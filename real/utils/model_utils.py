import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from typing import Optional, Tuple, Union

class CatLanguageModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.cat_config = config.get('cat_specific', {})
        
        # Add cat-specific prediction heads
        self.behavior_head = nn.Linear(config.n_embd, len(self.cat_config.get('behavior_categories', [])))
        self.personality_head = nn.Linear(config.n_embd, len(self.cat_config.get('personality_traits', [])))
        self.decision_head = nn.Linear(config.n_embd, len(self.cat_config.get('decision_making_factors', [])))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        behavior_labels: Optional[torch.LongTensor] = None,
        personality_labels: Optional[torch.LongTensor] = None,
        decision_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, dict]:
        # Get base model outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get the hidden states from the last layer
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, -1, :]  # Use last token for classification

        # Get predictions from cat-specific heads
        behavior_logits = self.behavior_head(pooled_output)
        personality_logits = self.personality_head(pooled_output)
        decision_logits = self.decision_head(pooled_output)

        # Calculate additional losses if labels are provided
        loss = outputs.loss if outputs.loss is not None else 0

        if behavior_labels is not None:
            behavior_loss = nn.CrossEntropyLoss()(behavior_logits, behavior_labels)
            loss = loss + behavior_loss

        if personality_labels is not None:
            personality_loss = nn.CrossEntropyLoss()(personality_logits, personality_labels)
            loss = loss + personality_loss

        if decision_labels is not None:
            decision_loss = nn.CrossEntropyLoss()(decision_logits, decision_labels)
            loss = loss + decision_loss

        return {
            'loss': loss,
            'language_model_logits': outputs.logits,
            'behavior_logits': behavior_logits,
            'personality_logits': personality_logits,
            'decision_logits': decision_logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

    def predict_cat_behavior(self, input_text: str, tokenizer) -> dict:
        """
        Predict cat behavior, personality traits, and decision factors for given input.
        """
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.forward(**inputs)

        behavior_probs = torch.softmax(outputs['behavior_logits'], dim=-1)
        personality_probs = torch.softmax(outputs['personality_logits'], dim=-1)
        decision_probs = torch.softmax(outputs['decision_logits'], dim=-1)

        return {
            'behaviors': behavior_probs.tolist()[0],
            'personality_traits': personality_probs.tolist()[0],
            'decision_factors': decision_probs.tolist()[0]
        } 