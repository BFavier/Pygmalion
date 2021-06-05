

def get_pretrained(model_name="bert-base-multilingual-cased"):
    from transformers import pipeline
    if language is None:
        model_name = "bert-base-multilingual-cased"
    else:
        model_name = f"bert-base-{language}-cased"
    model = pipeline("fill-mask", model=model_name)


def get_summarizer():
    from transformers import pipeline
    model = pipeline("summarization")
    return model


if __name__ == "__main__":
    import IPython
    IPython.embed()
