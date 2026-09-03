from kaggle.spaceshipTitanic.Preprocess import  Preprocess as ParentPreprocess

class Preprocess(ParentPreprocess):
    def __init__(self, relativePath, scaler, categoryData):
        super().__init__(relativePath, scaler, categoryData)