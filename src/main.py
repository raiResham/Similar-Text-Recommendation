from model import SimilarSentences


if __name__ == "__main__":
    simsen = SimilarSentences("paraphrase-MiniLM-L3-v2")
    top_n = 5
    more = "y"

    while True and more == "y":
        input_sentence = input("Please enter a sentence:")
        similar_sentences = simsen.similar_sentences(input_sentence, top_n)
        print(input_sentence)
        for sentence in similar_sentences:
            print(sentence)
        print("Do you want search more (y/n)?")
        more = input()

    
