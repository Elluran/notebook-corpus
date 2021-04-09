import json
import re
import sys
from io import StringIO
from urllib.request import urlretrieve
import pandas as pd
import fasttext


def clean_text(s):
    """prepares text for the classification"""
    s = s.lower()
    banned_symbols = {',', ':', '#', '.', '!',
                      '$'} | set(map(str, range(0, 10)))
    s = ''.join([ch for ch in s if ch not in banned_symbols])
    s = " ".join(s.split())
    if len(s) > 0 and s[-1] == ' ':
        s = s[:-1]
    if len(s) > 0 and s[0] == ' ':
        s = s[1:]
    return s

# hack disabling useless warning
old_stderr = sys.stderr
sys.stderr = StringIO()
lang_classifier = fasttext.load_model("lid.176.ftz")
# A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
# A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models
sys.stderr = old_stderr


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("no arguments provided")
        exit(0)
    filename = sys.argv[1]
    urlretrieve(
        f"https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/{filename}", filename)

    with open(filename) as notebook_file:
        notebook = json.load(notebook_file)

    print("nbformat:", notebook["nbformat"])
    metadata = notebook["metadata"]
    print("Python version:", metadata["language_info"]["version"])

    data = []

    for cell_id in range(len(notebook["cells"])):
        cell = notebook["cells"][cell_id]
        content = "".join(cell["source"])
        cell_type = cell["cell_type"]
        text_language = None
        function_calls = None

        if cell_type == "markdown":
            text_language = lang_classifier.predict(
                clean_text(content))[0][0][9:]
        elif cell_type == "code":
            function_calls = len(re.findall(r"\w\(", content))

        data.append([cell_id, cell_type, content,
                     text_language, function_calls])

    table = pd.DataFrame(data,
                         columns=["cell_id", "type", "content", "text_language", "function_calls"])

    table.to_csv('result.csv', index=False)
