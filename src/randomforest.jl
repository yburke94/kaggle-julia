using Images
using DataFrames
using DecisionTree

imageSize = 400 # 20x20 pixels
path = "../data/" # location of data files

function read_data(typeData, labels, imageSize, path)
  # Init empty matrix for all images
  x = zeros(size(labels, 1), imageSize)

  for (idx, idImage) in enumerate(labels["ID"])
      # Read image file
      nameFile = "$(path)/$(typeData)Resized/$(idImage).Bmp"
      img = imread(nameFile)

      # Convert image to float matrix
      temp = float32sc(img)

      # Convert colour images to greyscale by
      # taking average of the colour scales
      if ndims(temp) == 3
          temp = mean(temp.data, 1)
      end

      # Respape image to vector and add to matrix at row:idx.
      # Each row will be vector that represents every pixel in the image
      x[idx, :] = reshape(temp, 1, imageSize)
  end
  return x
end

# Load training + test data labels
trainingDataLabels = readtable("$(path)/trainLabels.csv")
testDataLabels = readtable("$(path)/sampleSubmission.csv")

# create matrix for training data
xTrain = read_data("train", trainingDataLabels, imageSize, path)
xTest = read_data("test", testDataLabels, imageSize, path)

# The labels are characters, but the algorithms recognize numbers,
# so we will map each character to an integer.
# Get only first character of string (convert from string to character).
# Apply the function to each element of the column "Class"
yTrain = map(x -> x[1], trainingDataLabels["Class"])
yTrain = int(yTrain) # Convert char to int for RandomForest alg.

# Random forest params
noOfFeaturesAtSplit = 20 # sqrt(number of features(400)).
noOfTrees = 50 # Larger is better but effects performance.
subsamplingRatio = 1.0 # Usually 1.0.

# Train random forest and get predictions on test data
model = build_forest(y_Train, x_Train, noOfFeaturesAtSplit, noOfTrees, subsamplingRatio)
predTest = apply_forest(model, xTest)

# Convert and save
testDataLabels["Class"] = char(predTest)
writetable("$(path)/juliaSubmission.csv", testDataLabels, seperator=',', header=true)
