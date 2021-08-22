import pandas as pd
from queue import PriorityQueue
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer



class Sentence:
    """
    A class used to represent a sentence, cosine similarity and its embedding.

    ...

    Attributes
    ----------
    cosine_sim : float
        A cosine similarity score

    sentence : str
        A sentence

    embedding : numpy.ndarray 
        An embedding of the sentence


    Methods
    -------
    __init__(cosine_sim, sentence, embedding)
        Creates and initializes Sentence object with cosine_sim, sentence and embedding

    __lt__(other) 
        Compares the cosine similarity scores and returns boolean value
    """

    def __init__(self, cosine_sim, sentence, embedding):
        self.cosine_sim = cosine_sim
        self.sentence = sentence
        self.embedding = embedding
    
    def __lt__(self, other):
        if self.cosine_sim <= other.cosine_sim:
            return True
        return False




class SimilarSentences:
    """
    A class used for sentence similarity.

    ...

    Attributes
    ----------
    sentences_list : list
        A list of Sentence object type

    model : sentence_transformers.SentenceTransformer.SentenceTransformer
        A sentence

    top_n_sentences : list
        A list of top n sentences
    
    priorityque  : PriorityQueue
        A priority queue for recommendation of sentences


    Methods
    -------
    load_dataset()
        Loads the sentences from datset

    get_embeddings(sentence) 
        Creates and returns the sentence embedding

    similar_sentences(sentence, top_n)
        Selects and returns top n sentences based on the sentence
    """

    sentences_list =[]

    def __init__(self, model_name):
        """
        Initializes and creates SimilarSentences class object and creates model.

        Parameters
        ----------
        model_name(str) : Name of a particular model
        """

        self.model_name = model_name
        # Create a transformer model specified by model_name
        self.model = SentenceTransformer(model_name)
        self.load_dataset()


    def load_dataset(self):
        """
        Loads sentence from dataset and adds into sentence_list

        Returns
        -------
        None
        """

        sts_df = pd.read_csv("../data/sts_test.csv")
        for idx, sentence in sts_df["sent_1"].iteritems():
            embedding = self.get_embeddings(sentence)
            SimilarSentences.sentences_list.append(Sentence(0, sentence, embedding))
            
    def get_embeddings(self, sentence):
        """
        Creates and returns the sentence embedding

        Parameters
        ----------
        sentence(str) : input sentence
        
        Returns
        -------
        embedding(numpy.ndarray) : embedding of the sentence 
        """
        
        embedding = self.model.encode(sentence)
        return embedding

    def similar_sentences(self, sentence, top_n):
        """
        Selects and returns top n sentences based on the sentence.

        Parameters
        ----------
        sentence(str) : input sentence
        top_n(int) : number of top similar sentences

        Returns
        -------
        top_n_sentences(list) : list of top n sentences based on the input sentence
        """
        
        self.top_n_sentences = []
        self.priorityque = PriorityQueue()
        embedding = self.get_embeddings(sentence)
        for sentence in SimilarSentences.sentences_list:
            # Calculating cosine similarity
            cosine_sim =  1 - distance.cosine(embedding, sentence.embedding)
            # Update cosine value in current sentence object
            # We are negating the cosine_sim as priority queue gives min value by default
            sentence.cosine_sim = -cosine_sim
            # Add this sentence object to priority queue
            self.priorityque.put(sentence)

        # Find top n similar sentences
        while top_n > 0:
            self.sentence_obj = self.priorityque.get()
            self.sentence = self.sentence_obj.sentence
            # Negating the cosine_sim as we have made in negative above to use it in priority queue
            self.cosine_sim = -self.sentence_obj.cosine_sim
            self.top_n_sentences.append((self.cosine_sim, self.sentence))
            top_n -= 1
        # Make PriorityQueue object garbage collectable.
        self.priorityque = None   
        return self.top_n_sentences