from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.models.clip.configuration_clip import CLIPTextConfig, CLIPConfig, CLIPVisionConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import _expand_mask, CLIPTextEmbeddings, CLIPEncoder, CLIPPreTrainedModel, CLIPOutput,CLIPTextTransformer, CLIPVisionTransformer


# code taken from hugging face transformers library
# https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/models/clip/modeling_clip.py
# modified lines are marked 

class ModifiedCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

# MODIFIED
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_ids is None:
            input_shape = inputs_embeds.size() 
            bsz, seq_len, dim = input_shape
        else:
            input_shape = input_ids.size()
            bsz, seq_len = input_shape
            input_ids = input_ids.view(-1, input_shape[-1])


        hidden_states = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds )
##########

        
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = self._build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
            hidden_states.device
        )
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = None

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _build_causal_attention_mask(self, bsz, seq_len, dtype):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
        mask.fill_(torch.tensor(torch.finfo(dtype).min))
        mask.triu_(1)  # zero out the lower diagonal
        mask = mask.unsqueeze(1)  # expand mask
        return mask



class ModifiedCLIPTextModel(CLIPPreTrainedModel):
    config_class = CLIPTextConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPTextConfig):
        super().__init__(config)
# MODIFIED
        self.text_model = ModifiedCLIPTextTransformer(config)
##########
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import CLIPTokenizer, CLIPTextModel
        >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        
        
# MODIFIED
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inputs_embeds=inputs_embeds, 
        )
##########






# class ModifiedCLIPModel(CLIPPreTrainedModel):
#     config_class = CLIPConfig

#     def __init__(self, config: CLIPConfig):
#         super().__init__(config)

#         if not isinstance(config.text_config, CLIPTextConfig):
#             raise ValueError(
#                 "config.text_config is expected to be of type CLIPTextConfig but is of type"
#                 f" {type(config.text_config)}."
#             )

#         if not isinstance(config.vision_config, CLIPVisionConfig):
#             raise ValueError(
#                 "config.vision_config is expected to be of type CLIPVisionConfig but is of type"
#                 f" {type(config.vision_config)}."
#             )

#         text_config = config.text_config
#         vision_config = config.vision_config

#         self.projection_dim = config.projection_dim
#         self.text_embed_dim = text_config.hidden_size
#         self.vision_embed_dim = vision_config.hidden_size

#         self.text_model = ModifiedCLIPTextTransformer(text_config)
#         self.vision_model = CLIPVisionTransformer(vision_config)

#         self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
#         self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
#         self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_text_features(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> torch.FloatTensor:
#         r"""
#         Returns:
#             text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
#             applying the projection layer to the pooled output of [`CLIPTextModel`].
#         Examples:
#         ```python
#         >>> from transformers import CLIPTokenizer, CLIPModel
#         >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#         >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
#         >>> text_features = model.get_text_features(**inputs)
#         ```"""
#         # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         text_outputs = self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

# #         TODO: fail pooled output je None
#         pooled_output = text_outputs[1]
#         text_features = self.text_projection(pooled_output)

#         return text_features

#     def get_image_features(
#         self,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> torch.FloatTensor:
#         r"""
#         Returns:
#             image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
#             applying the projection layer to the pooled output of [`CLIPVisionModel`].
#         Examples:
#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import CLIPProcessor, CLIPModel
#         >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)
#         >>> inputs = processor(images=image, return_tensors="pt")
#         >>> image_features = model.get_image_features(**inputs)
#         ```"""
#         # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         vision_outputs = self.vision_model(
#             pixel_values=pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = vision_outputs[1]  # pooled_output
#         image_features = self.visual_projection(pooled_output)

#         return image_features

# # MODIFIED

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         pixel_values: Optional[torch.FloatTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         return_loss: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#     ) -> Union[Tuple, CLIPOutput]:
#         r"""
#         Returns:
#         Examples:
#         ```python
#         >>> from PIL import Image
#         >>> import requests
#         >>> from transformers import CLIPProcessor, CLIPModel
#         >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#         >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#         >>> image = Image.open(requests.get(url, stream=True).raw)
#         >>> inputs = processor(
#         ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
#         ... )
#         >>> outputs = model(**inputs)
#         >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#         >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
#         ```"""
#         # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         vision_outputs = self.vision_model(
#             pixel_values=pixel_values,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         text_outputs = self.text_model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             inputs_embeds=inputs_embeds, 

#         )
        
#         image_embeds = vision_outputs[1]

#         image_embeds = self.visual_projection(image_embeds)

#         text_embeds = text_outputs[0].squeeze(0)
#         text_embeds = self.text_projection(text_embeds)

#         # normalized features
#         image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
#         text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        

#         # cosine similarity as logits
#         logit_scale = self.logit_scale.exp()
#         logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
#         print(logits_per_text.shape)
#         logits_per_image = logits_per_text.t()

#         loss = None
#         if return_loss:
#             loss = clip_loss(logits_per_text)

#         if not return_dict:
#             output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
#             return ((loss,) + output) if loss is not None else output

#         return CLIPOutput(
#             loss=loss,
#             logits_per_image=logits_per_image,
#             logits_per_text=logits_per_text,
#             text_embeds=text_embeds,
#             image_embeds=image_embeds,
#             text_model_output=text_outputs,
#             vision_model_output=vision_outputs,
#         )
# ##########