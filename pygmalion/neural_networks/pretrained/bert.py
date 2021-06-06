

def get_pretrained(model_name="bert-base-multilingual-cased"):
    from transformers import pipeline
    model = pipeline("fill-mask", model=model_name)
    return model


def get_summarizer():
    from transformers import pipeline
    model = pipeline("summarization")
    return model


if __name__ == "__main__":
    import IPython
    IPython.embed()
