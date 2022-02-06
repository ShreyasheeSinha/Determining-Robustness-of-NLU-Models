from transformers import AutoTokenizer, AutoModelForSequenceClassification

class Transformer():

    def __init__(self, model_name, classification_head=False):
        if classification_head:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels = 3, # The number of output labels--3 for entailment classification tasks.  
                output_attentions = False, # Whether the model returns attentions weights.
                output_hidden_states = False, # Whether the model returns all hidden-states.
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_model_and_tokenizer(self):
        return self.model, self.tokenizer