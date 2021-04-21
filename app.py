from flask import Flask, jsonify, request
from pororo import Pororo
from google.cloud import language_v1
import os
import json
import numpy as np
from google.protobuf.json_format import MessageToJson

app = Flask(__name__)
debug = True if os.getenv("FLASK_ENV") != 'production' else False
credential_path = "assi-test-prod-065ee6d6e726.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

@app.route('/kakao/review', methods=['POST'])
def review():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    review = Pororo(task="review", lang=lang)
    result = review(text)
    return jsonify({"kakao": {"review": result}})


@app.route('/kakao/paraphrase_identification', methods=['POST'])
def paraphrase_identification():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    paraphrase_identification = Pororo(
        task="paraphrase_identification", lang=lang)
    result = paraphrase_identification(text)
    return jsonify({"kakao": {"paraphrase_identification": result}})


@app.route('/kakao/sentence_embedding', methods=['POST'])
def sentence_embedding():
    data = request.get_json()
    cands = data.get("cands", "")
    query = data.get("query", "")
    lang = data.get("lang", "")
    sentence_embedding = Pororo(
        task="sentence_embedding", lang=lang)
    result = sentence_embedding.find_similar_sentences(query, cands)
    return jsonify({"kakao": {"sentence_embedding": result}})


@app.route('/kakao/natural_language_inference', methods=['POST'])
def natural_language_inference():
    data = request.get_json()
    sentenceOne = data.get("sentenceOne", "")
    sentenceTwo = data.get("sentenceTwo", "")
    lang = data.get("lang", "")
    natural_language_inference = Pororo(task="nli", lang=lang)
    result = natural_language_inference(sentenceOne, sentenceTwo)
    return jsonify({"kakao": {"natural_language_inference": result}})


@app.route('/kakao/sementic_textual_similarity', methods=['POST'])
def sementic_textual_similarity():
    data = request.get_json()
    sentenceOne = data.get("sentenceOne", "")
    sentenceTwo = data.get("sentenceTwo", "")
    lang = data.get("lang", "")
    sementic_textual_similarity = Pororo(task="similarity", lang=lang)
    sts = Pororo(task="similarity", lang="ko")
    result = sementic_textual_similarity(sentenceOne, sentenceTwo)
    return jsonify({"kakao": {"sementic_textual_similarity": result}})


@app.route('/kakao/zero_shot_topic_classification', methods=['POST'])
def zero_shot_topic_classification():
    data = request.get_json()
    text = data.get("text", "")
    categories = data.get("categories", "")
    lang = data.get("lang", "")
    zero_shot_topic_classification = Pororo(task="zero-topic", lang=lang)
    result = zero_shot_topic_classification(text, categories)
    return jsonify({"kakao": {"zero_shot_topic_classification": result}})


@app.route('/kakao/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    sentiment_analysis = Pororo(task="sentiment", lang=lang)
    result = sentiment_analysis(text)
    return jsonify({"kakao": {"sentiment_analysis": result}})


@app.route('/kakao/contextualized_embedding', methods=['POST'])
def contextualized_embedding():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    contextualized_embedding = Pororo(task="cse", lang=lang)
    result = contextualized_embedding(text)
    return jsonify({"kakao": {"contextualized_embedding": result.tolist()}})


@app.route('/kakao/dependency_parsing', methods=['POST'])
def dependency_parsing():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    dependency_parsing = Pororo(task="dep_parse", lang=lang)
    result = dependency_parsing(text)
    print(result)
    return jsonify({"kakao": {"dependency_parsing": result}})


@app.route('/kakao/fill_in_the_blank', methods=['POST'])
def fill_in_the_blank():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    fill_in_the_blank = Pororo(task="fib", lang=lang)
    result = fill_in_the_blank(text)
    return jsonify({"kakao": {"fill_in_the_blank": result.tolist()}})


@app.route('/kakao/machine_reading_comprehension', methods=['POST'])
def machine_reading_comprehension():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    machine_reading_comprehension = Pororo(task="mrc", lang=lang)
    result = machine_reading_comprehension(text)
    return jsonify({"kakao": {"machine_reading_comprehension": result.tolist()}})


@app.route('/kakao/named_entity_recognition', methods=['POST'])
def named_entity_recognition():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    named_entity_recognition = Pororo(task="mrc", lang=lang)
    result = named_entity_recognition(text)
    return jsonify({"kakao": {"named_entity_recognition": result.tolist()}})
    
@app.route('/kakao/text_summarization', methods=['POST'])
def text_summarization():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    text_summarization = Pororo(task="summarization", model="abstractive", lang=lang)
    result = text_summarization(text)
    return jsonify({"kakao": {"text_summarization": result}})

@app.route('/kakao/translation', methods=['POST'])
def translation():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    translation = Pororo(task="translation", lang="multi")
    result = translation(text, src="ko", tgt="en")
    return jsonify({"kakao": {"translation": result}})

@app.route('/kakao/ocr', methods=['POST'])
def ocr():
    data = request.get_json()
    file = data.get("file", "")
    lang = data.get("lang", "")
    ocr = Pororo(task="ocr", lang=lang)
    result = ocr(file)
    return jsonify({"kakao": {"ocr": result}})


@app.route('/kakao/word_sense_disambiguation', methods=['POST'])
def word_sense_disambiguation():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    word_sense_disambiguation = Pororo(task="wsd", lang=lang)
    result = word_sense_disambiguation(text)
    return jsonify({"kakao": {"word_sense_disambiguation": result}})

@app.route('/google/sample_analyze_sentiment', methods=['POST'])
# 감정 분석하기 sentiment score가 높을 수록 긍정적인 것이다. 
def sample_analyze_sentiment():
    credential_path = "assi-test-prod-065ee6d6e726.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    result = {}
    client = language_v1.LanguageServiceClient()
    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT
    document = {"content": text, "type_": type_, "language": lang}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Get overall sentiment of the input document
    # print(u"Document sentiment score: {}".format(response.document_sentiment.score))
    result['sentiment_score'] = response.document_sentiment.score

    result['sentiment_magnitude'] = response.document_sentiment.magnitude
    # Get sentiment for all sentences in the document

    result_texts = []
    for sentence in response.sentences:
        result_text = {}
        print(u"Sentence text: {}".format(sentence.text.content))
        result_text["sentence_content"] = sentence.text.content
        print(u"Sentence sentiment score: {}".format(sentence.sentiment.score))
        result_text["sentence_score"] = sentence.sentiment.score
        print(u"Sentence sentiment magnitude: {}".format(sentence.sentiment.magnitude))
        result_text["sentence_magnitude"] = sentence.sentiment.magnitude
        result_texts.append(result_text)
    result['sentences']= result_texts
    return jsonify({"google": result})


@app.route('/google/sample_analyze_entities', methods=['POST'])
def sample_analyze_entities():
    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    result = {}
    client = language_v1.LanguageServiceClient()
    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT
    # https://cloud.google.com/natural-language/docs/languages
    language = "ko"
    document = {"content": text, "type_": type_, "language": lang}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_entities(request = {'document': document, 'encoding_type': encoding_type})

    # Loop through entitites returned from the API
    result_texts = []
    for entity in response.entities:
        result_text = {}
        print(u"Representative name for the entity: {}".format(entity.name))
        result_text["representative_entitiy_name"] = entity.name
        # Get entity type, e.g. PERSON, LOCATION, ADDRESS, NUMBER, et al
        print(u"Entity type: {}".format(language_v1.Entity.Type(entity.type_).name))
        result_text["entitiy_type"] = language_v1.Entity.Type(entity.type_).name
        # Get the salience score associated with the entity in the [0, 1.0] range
        print(u"Salience score: {}".format(entity.salience))
        #단어가 나타내는 중요성 수치
        result_text["salience_score"] = entity.salience
        metadatas = []
        for metadata_name, metadata_value in entity.metadata.items():
            metadata_items = {}
            print(u"{}: {}".format(metadata_name, metadata_value))
            metadata_items["metadata_name"] = metadata_name
            metadata_items["metadata_value"] = metadata_value
            metadatas.append(metadata_items)
            result_text["metadatas"] = metadatas
        # Loop over the mentions of this entity in the input document.
        # The API currently supports proper noun mentions.
        mentions = []
        for mention in entity.mentions:
            mention_items = {}
            print(u"Mention text: {}".format(mention.text.content))
            mention_items["mention_content"] = mention.text.content
            # Get the mention type, e.g. PROPER for proper noun
            print(
                u"Mention type: {}".format(language_v1.EntityMention.Type(mention.type_).name)
            )
            mention_items["mention_type"] = language_v1.EntityMention.Type(mention.type_).name
            mentions.append(metadata_items)
            result_text["mentions"] = mentions
        result_texts.append(result_text)
    result['entities']= result_texts
    return jsonify({"google": result})


@app.route('/google/sample_analyze_syntax', methods=['POST'])
def sample_analyze_syntax():

    data = request.get_json()
    text = data.get("text", "")
    lang = data.get("lang", "")
    result = {}
    client = language_v1.LanguageServiceClient()

    # text_content = 'This is a short sentence.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    document = {"content": text, "type_": type_, "language": lang}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_syntax(request = {'document': document, 'encoding_type': encoding_type})
    # Loop through tokens returned from the API
    result_texts = []
    for token in response.tokens:
        result_text = {}
        # Get the text content of this token. Usually a word or punctuation.
        text = token.text
        print(u"Token text: {}".format(text.content))
        result_text["token_text_content"] = text.content
        print(
            u"Location of this token in overall document: {}".format(text.begin_offset)
        )
        result_text["token_text_begin_offset"] = text.begin_offset
        # Get the part of speech information for this token.
        # Parts of spech are as defined in:
        # http://www.lrec-conf.org/proceedings/lrec2012/pdf/274_Paper.pdf
        part_of_speech = token.part_of_speech
        # Get the tag, e.g. NOUN, ADJ for Adjective, et al.
        print(
            u"Part of Speech tag: {}".format(
                language_v1.PartOfSpeech.Tag(part_of_speech.tag).name
            )
        )
        result_text["part_of_speech_name"] = language_v1.PartOfSpeech.Tag(part_of_speech.tag).name
        # Get the voice, e.g. ACTIVE or PASSIVE
        print(u"Voice: {}".format(language_v1.PartOfSpeech.Voice(part_of_speech.voice).name))
        # Get the tense, e.g. PAST, FUTURE, PRESENT, et al.
        result_text["part_of_speech_voice"] = language_v1.PartOfSpeech.Voice(part_of_speech.voice).name
        print(u"Tense: {}".format(language_v1.PartOfSpeech.Tense(part_of_speech.tense).name))
        result_text["tense"] = language_v1.PartOfSpeech.Tense(part_of_speech.tense).name
        # See API reference for additional Part of Speech information available
        # Get the lemma of the token. Wikipedia lemma description
        # https://en.wikipedia.org/wiki/Lemma_(morphology)
        print(u"Lemma: {}".format(token.lemma))
        result_text["lemma"] = token.lemma
        # Get the dependency tree parse information for this token.
        # For more information on dependency labels:
        # http://www.aclweb.org/anthology/P13-2017
        dependency_edge = token.dependency_edge
        print(u"Head token index: {}".format(dependency_edge.head_token_index))
        result_text["head_token_index"] = dependency_edge.head_token_index
        print(
            u"Label: {}".format(language_v1.DependencyEdge.Label(dependency_edge.label).name)
        )
        result_text["label"] = language_v1.DependencyEdge.Label(dependency_edge.label).name
        result_texts.append(result_text)
    result["tokens"] = result_texts
    return jsonify({"google": result})
if __name__ == "__main__":
    app.run(host="127.0.0.1" if debug else "0.0.0.0", port=8000, debug=debug)
