### This is a Spam Detection Filter using Python and Spark.
### We will NLP technique to filter out spam message based on the dataset provided.
### NaivBayes model will be used for classification.

from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # Create a session
    spark = SparkSession.builder.appName('nlp').getOrCreate()

    # Retrieve dataset
    data = spark.read.csv('smsspamcollection/SMSSpamCollection', inferSchema=True, sep='\t')

    # format data so that it's more readable
    data = data.withColumnRenamed('_co', 'class').withColumnRenamed('_c1', 'text')

    # clean up data using NLP tool
    #
    data = data.withColumn('length', length(data['text']))

    # data.group('class').mean().show()
    # Expected result:
    # +-----+-----------------+
    # |class|      avg(length)|
    # +-----+-----------------+
    # |  ham|71.48663212435233|
    # | spam|138.6706827309237|
    # +-----+-----------------+
    #
    # The average length of a spam message is much longer so it can already see taht there's some
    # sort of a ham column versus spam column. Meaning this may be a useful feature and this is
    # essentially feature engineering that you take a text which is your raw feature and your feature
    # engieers some other column feature. In this case, the actual length of that text.

    tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
    stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
    count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
    idf = IDF(inputCol='c_vec', outputCol='tf_idf')
    ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')

    # Clean up our dataset using above tokenization
    clean_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

    # Build a NaivBayes model
    model = NaiveBayes()

    # Create a pipeline
    data_prep_pipe = Pipeline(stages=[ham_spam_to_numeric, tokenizer, stop_remove, count_vec, idf, clean_up])

    # Fit pipeline to the actual dataset
    cleaner = data_prep_pipe.fit(data)

    # Final clean data
    clean_data = cleaner.tramsform(data)

    # Training
    #
    # We are only interested in label and features
    clean_data = clean_data.select('label', 'features')

    # Split dataset
    training, test = clean_data.randomSplit([0.7, 0.3])

    # Fit training data
    spam_detector = model.fit(training)

    # test result
    test_results = spam_detector.transform(test)

    # show test result
    test_results.show()

    # Compare label verus prediction using evaluation metircs which is multiclass classification evaluator in this case
    accuracy_evaluator = MulticlassClassificationEvaluator()
    accuracy = accuracy_evaluator.evaluate(test_results)

    # Show result
    print('Accuray of Naivbayes Model')
    print(accuracy)      # My Result = 0.91871286095523


if __name__ == "__main__":
    main()