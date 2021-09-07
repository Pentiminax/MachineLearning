using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML;

namespace MachineLearning
{
    class Program
    {
        private static MLContext context;
        private static ITransformer model;
        private static readonly string dataPath = Path.Combine(Environment.CurrentDirectory, "data", "train.txt");

        static void Main(string[] args)
        {
            context = new();

            model = GetModel();

            UseModel();
        }

        public static ITransformer BuildAndTrainModel(IDataView data)
        {
            var estimator = context.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            Console.WriteLine("Création et entrainement sur du modèle");

            var model = estimator.Fit(data);

            Console.WriteLine("Fin de l'entrainement");

            return model;
        }

        public static void Evaluate(IDataView data)
        {
            Console.WriteLine("Évaluation de la précision modèle");

            var predictions = model.Transform(data);
            var metrics = context.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"Précision : {metrics.Accuracy:P2}");

            Console.WriteLine("Fin de l'évaluation du modèle");
        }

        public static ITransformer GetModel()
        {
            if (File.Exists("model.zip"))
            {
                return context.Model.Load("model.zip", out DataViewSchema schema);
            }

            var data = context.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, allowQuoting: true);
            var splitDataView = context.Data.TrainTestSplit(data);

            model = BuildAndTrainModel(splitDataView.TrainSet);

            Evaluate(data);

            context.Model.Save(model, data.Schema, "model.zip");

            return model;
        }

        public static void UseModel()
        {
            PredictionEngine<SentimentData, SentimentPrediction> predictionEngine =
                context.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Une fin bâclée extrêmement décevante avec un scénario qui ne pas pas sense. Zéro adrénaline après 10 ans de série haletante... Bref une fin vraiment nul !"
                },
                new SentimentData
                {
                    SentimentText = "La pire fin de série !!! Pourquoi gâcher un chef d'oeuvres... en fin pourrie !!! De semaine en semaine les épisodes mon déçus c'était plat et bâclé... la mort de certains personnages son nul et incompréhensible !! Déçu déçu déçu !!!! La pire fin de tout les temps"
                },
                new SentimentData
                {
                    SentimentText = "Cette série a marqué toute une décennie et une génération. Un seul mot peut véritablement la qualifier: magistrale."
                }
            };

            var reviews = context.Data.LoadFromEnumerable(sentiments);
            var predictions = model.Transform(reviews);
            var predictionResults = context.Data.CreateEnumerable<SentimentPrediction>(predictions, reuseRowObject: true);

            Console.WriteLine("Test de prédiction");

            foreach (var prediction in predictionResults)
            {
                var predictionText = prediction.Prediction ? "Critique positive" : "Critique négative";
                Console.WriteLine($"Critique : {prediction.SentimentText}");
                Console.WriteLine($"Prédiction : {predictionText}");
                Console.WriteLine($"Probabilité : {prediction.Probability:P2}");
                Console.WriteLine("--------------------------------------------------------------------------------");
            }
        }
    }
}
