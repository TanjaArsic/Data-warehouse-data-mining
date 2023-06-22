from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, col, when, lit
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes, LinearSVC, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# postavlja konfiguraciju SparkSession objekta za lokalni način rada (1 komp)
spark = SparkSession.builder \
    .master("local") \
    .appName("StrokePrediction") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
dataset_df = spark.read.csv(
    'healthcare-dataset-stroke-data.csv', inferSchema=True, header=True)

dataset_df.printSchema()
dataset_df.describe().show(2)  # sveukupna statistika, prva 2 reda

# željene kategoričke vrednosti prikazuje, gleda se u ovom slučaju po vrednostima stroke kolko ih ima, tj kolko njih ima i kolko njih nema šlog
dataset_df.groupby("stroke").count().show()
# nebalansiran set podataka

# Sredjuju se podaci
dataset_df = dataset_df.drop('id')  # izbacuje se id

mean_bmi = dataset_df.select(mean(col('bmi'))).first()[
    0]  # uzimam srednju vrednost bmi iz kolone

dataset_df = dataset_df.withColumn('bmi', when(
    col('bmi') == "N/A", mean_bmi).otherwise(col('bmi')))  # menjaj svuda gde je "N/A"

dataset_df = dataset_df.withColumn("bmi", dataset_df["bmi"].cast(
    "double"))  # bmi je bio string sad je double

dataset_df.show(15)

# izdvoji kolone sa string tipom podataka
string_columns = [col for col, dtype in dataset_df.dtypes if dtype == 'string']
encoded_cols = list()
stages = list()
# Rade se StringIndexer i OneHotEncoder faze
for column in string_columns:
    # MNOBO JE VAŽNO OVO INVALID DA HENDLUJE JER SAM SE NAMUČILAAAAAAA BEZ TOGA
    indexer = StringIndexer(
        inputCol=column, outputCol=column + "_index", handleInvalid="keep")
    encoder = OneHotEncoder(
        inputCols=[column + "_index"], outputCols=[column + "_encoded"])
    stages += [indexer, encoder]

# Create a Pipeline with the stages, ovo je s chatgpt jer moj kod nije hteo da radi bez pipeline :DDDDDD
pipeline = Pipeline(stages=stages)

# Fit and transform the pipeline on the DataFrame (gle ovde transform oce da radi :DDDDDDDDDDDDDDD)
encoded_df = pipeline.fit(dataset_df).transform(dataset_df)
encoded_df.show(10)

# iterira se kroz sve string kolone i menja se originalno s enkodiranim vrednostima, a index vrednosti se dropuju sta ce nam
for column in string_columns:
    encoded_col = column + "_encoded"
    index_col = column + "_index"
    mapping = encoded_df.select(column, index_col).distinct().collect()
    print(f"Vrednosti u koloni '{column}' se preslikavaju:")
    for row in mapping:
        print(f"{row[column]} -> {row[index_col]}")
    encoded_df = encoded_df.withColumn(column, col(
        encoded_col)).drop(encoded_col).drop(index_col)
dataset_df = encoded_df
dataset_df.show(10)

# ====gender====
# male je 1.0
# female je 0.0
# other je 2.0

# ====ever_married====
# married yes je 0.0
# no je 1.0

# ====smoking_status====
# formerly smoked je 2.0
# never smoked je 0.0
# smokes je 3.0
# unknown je 1.0

# ====work_type====
# private 0.0
# self-employed 1.0
# children  2.0
# Never_worked  4.0
# govt_job -> 3.0

# ====residence_type====
# urban 0.0
# rural 1.0


# kako se čita (2, [1], [1.0]): 2 je dužina vektora, 1 označava indeks gde postoji nenulta vrednost (svi ostali indeksi su 0), 1.0 je ta vrednost na indeksu 1

# jubilarna linija koda da se počne s treniranjem modela
# definišu se svi atributi koji ulaze u klasifikaciju
feature_columns = ["gender", "age", "hypertension", "heart_disease", "ever_married",
                   "work_type", "Residence_type", "avg_glucose_level", "bmi",
                   "smoking_status"]

# kreiranje VectorAssemblera
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
dataset_df = assembler.transform(encoded_df)

# pravi se lista gde se instanciraju modeli (ovde mogu i svi ostali koji će da se koriste)
classifiers = [
    LogisticRegression(featuresCol="features", labelCol="stroke"),
    NaiveBayes(featuresCol="features", labelCol="stroke"),
    LinearSVC(featuresCol="features", labelCol="stroke"),
    DecisionTreeClassifier(featuresCol="features", labelCol="stroke"),
    RandomForestClassifier(featuresCol="features", labelCol="stroke")
]

for classifier in classifiers:
    print(
        f'Klasifikator {classifier.__class__.__name__} s 30:70 test:trening podelom podataka:')

    train_data, test_data = dataset_df.randomSplit([0.7, 0.3], seed=42)

    # model se pravi na osnovu trening podataka
    model = classifier.fit(train_data)
    # prave se predikcije s test podacima
    predictions = model.transform(test_data)

    predicted_strokes = np.array(predictions.select(
        "prediction").rdd.flatMap(lambda x: x).collect())  # predikcije
    
    actual_strokes = np.array(predictions.select("stroke").rdd.flatMap(
        lambda x: x).collect())  # ekšual stronks (desilo se)

    print(
        f'Evaluacija {classifier.__class__.__name__} klasifikatora za 30:70 test:trening podelu podataka:')
    print(classification_report(actual_strokes, predicted_strokes))
    print("Matrica konfuzije:")
    print(confusion_matrix(actual_strokes, predicted_strokes))

    evaluator = BinaryClassificationEvaluator(labelCol="stroke")
    auc = evaluator.evaluate(predictions)
    print("Površina ispod ROC krive (AUC):", auc)
    print("------------------------------------")


for classifier in classifiers:
    print(f'Klasifikator {classifier.__class__.__name__} sa cross-validation:')

# Ovde se pomocu grida testiraju najbolji parametri za konkretni klasifikator, ima raznih

    # smoothing se koristi kako bi se izbeglo da model dodeli nultu verovatnoću nekom atributu koji se nije pojavio u trening skupu podataka
    if isinstance(classifier, NaiveBayes):
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.smoothing, [0.1, 0.5, 1.0]) \
            .build()
        
    elif isinstance(classifier, DecisionTreeClassifier):
        # MaxDepth je maksimalna dubina stabla koju dozvoljava Decision Tree algoritam
        # MaxBins predstavlja maksimalni broj binova koji će se koristiti prilikom diskretizacije numeričkih atributa u Decision Tree algoritmu
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.maxDepth, [2, 5, 10]) \
            .addGrid(classifier.maxBins, [10, 20, 30]) \
            .build()

    elif isinstance(classifier, RandomForestClassifier):
        # NumTrees je broj stabala koji će biti konstruisani u Random Forest algoritmu
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.numTrees, [50, 100, 150]) \
            .addGrid(classifier.maxDepth, [2, 5, 10]) \
            .build()
    else:
        # regParam kontroliše jačinu regularizacije (manje vrednosti regParam smanjuju regularizaciju, dok veće vrednosti je pojačavaju), što znači da smanjuje overfitting
        # MaxIter je maksimalni broj iteracija koje će alg. izvršiti prilikom učenja modela
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.regParam, [0.01, 0.1, 1.0]) \
            .addGrid(classifier.maxIter, [10, 20, 30]) \
            .build()

    crossval = CrossValidator(estimator=classifier,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(
                                  labelCol="stroke"),
                              numFolds=5)

    cv_model = crossval.fit(dataset_df)  # kreira model
    best_model = cv_model.bestModel  # uzima najbolji
    predictions = best_model.transform(dataset_df)

    predicted_strokes = np.array(predictions.select(
        "prediction").rdd.flatMap(lambda x: x).collect())  # predikcije
    
    actual_strokes = np.array(predictions.select("stroke").rdd.flatMap(
        lambda x: x).collect())  # desilose

    print(
        f'Evaluacija {classifier.__class__.__name__} klasifikatora za cross-validation podelu podataka:')
    print(classification_report(actual_strokes, predicted_strokes))
    print("Matrica konfuzije:")
    print(confusion_matrix(actual_strokes, predicted_strokes))

    evaluator = BinaryClassificationEvaluator(labelCol="stroke")
    auc = evaluator.evaluate(predictions)
    print("Površina ispod ROC krive (AUC):", auc)
    print("------------------------------------")
