using Microsoft.ML;
using Microsoft.ML.Vision;

//https://www.youtube.com/watch?v=ppRauvf6uCs

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var imageFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "flowers");
            var files = Directory.GetFiles(imageFolder, "*", SearchOption.AllDirectories);
            var images = files.Select(file => new ImageData
            {
                ImagePath = file,
                Label = Directory.GetParent(file).Name
            });

            var context = new MLContext();
            var imageData = context.Data.LoadFromEnumerable(images);
            var imageDataShuffled = context.Data.ShuffleRows(imageData);

            var testTrainData = context.Data.TrainTestSplit(imageDataShuffled, testFraction: 0.2);

            var validationData = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", 
                keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes("Image", imageFolder, "ImagePath"))
                .Fit(testTrainData.TestSet)
                .Transform(testTrainData.TestSet);

            var imagePipeline = context.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: Microsoft.ML.Transforms.ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(context.Transforms.LoadRawImageBytes("Image", imageFolder, "ImagePath"));

            var imagesDataModel = imagePipeline.Fit(testTrainData.TrainSet);
            var imageDataView = imagesDataModel.Transform(testTrainData.TrainSet);

            var options = new ImageClassificationTrainer.Options()
            {
                Arch = ImageClassificationTrainer.Architecture.ResnetV250,
                Epoch = 100,
                BatchSize = 20,
                LearningRate = 0.01f,
                LabelColumnName = "LabelKey",
                FeatureColumnName = "Image",
                ValidationSet = validationData
            };

            var pipeline = context.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(imageDataView);

            var predictionEngine = context.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(model);

            var testImagesFolder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "test1");
            var testFiles = Directory.GetFiles(testImagesFolder, "*", SearchOption.AllDirectories);
            var testImages = testFiles.Select(file => new ImageModelInput
            {
                ImagePath = file,
            });
            Console.WriteLine(Environment.NewLine);

            var testImageData = context.Data.LoadFromEnumerable(testImages);

            var testImageDataView = imagePipeline.Fit(testImageData).Transform(testImageData);
            var predictions = model.Transform(testImageDataView);

            var testPredictions = context.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false);

            foreach(var prediction in testPredictions)
            {
                Console.WriteLine($"Image: {Path.GetFileName(prediction.ImagePath)}, Predicted Label: {prediction.PredictedLabel}");
            }
        }
    }
}