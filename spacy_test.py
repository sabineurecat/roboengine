import spacy
import pdb

# 加载英语模型

def extract_noun_phrases_with_adjectives(text, nlp):
    doc = nlp(text)
    noun_phrases = []
    for nph in doc.noun_chunks:
        adjectives = [token.text for token in nph if token.pos_ == "ADJ"]
        noun_phrases.append(nph.text)
        # if adjectives:
        #     noun_phrases.append((nph.text, adjectives))
        # else:
        #     noun_phrases.append((nph.text, None))
    return noun_phrases



text = "lift the knife"
# extract_noun_phrases_with_adjectives(text)

# 提取名词短语和形容词
nlp = spacy.load("en_core_web_sm")
noun_phrases_with_adjectives = extract_noun_phrases_with_adjectives(text, nlp)

print(noun_phrases_with_adjectives)
# # 输出结果
# for np, adj in noun_phrases_with_adjectives:
#     print(f"名词短语: {np}, 形容词: {adj}")

pdb.set_trace()