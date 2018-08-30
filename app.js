class Classification {
  constructor(trainFilename, testFilename) {
    this.tf = require('@tensorflow/tfjs');
    require('@tensorflow/tfjs-node'); 
    let fs = require('fs')
    let path = require('path')
    let trainFile = fs.readFileSync(path.join(__dirname, trainFilename), 'utf8')
    let testFile = fs.readFileSync(path.join(__dirname, testFilename), 'utf8')
    this.prepareTrainingSet(trainFile)
    this.prepareTestingSet(testFile)
  }

  prepareTrainingSet(trainFile) {
    const inputValues = []
    const outputValues = []
    const dataOfArray = trainFile.split('\r\n')
    console.log(dataOfArray)
    dataOfArray.forEach(row => {
      const arrayRow = row.split(',')
      const out = arrayRow.pop()
      switch (out) {
        case 'Iris-setosa':
        outputValues.push([0, 0, 1])
        break
        case 'Iris-versicolor':
        outputValues.push([0, 1, 0])
        break
        case 'Iris-virginica':
        outputValues.push([1, 0, 0])
        break
      }
      inputValues.push(arrayRow.map(el => Number(el)))
      console.log(this.inputShapeNumber)
      this.inputShapeNumber = inputValues[0].length
      this.outputShapeNumber = outputValues[0].length
    });
    const input = this.tf.tensor2d(inputValues)
    const output = this.tf.tensor2d(outputValues)
    this.buildModel(input, output)
  }

  prepareTestingSet(testFile) {
    const testValues = []
    const dataOfArray = testFile.split('\n')
    dataOfArray.forEach(row => {
      const arrayRow = row.split(',')
      arrayRow.pop()
      testValues.push(arrayRow)
    });
    this.test = this.tf.tensor2d(testValues)
  }

  buildModel(input, output) {
    const model = this.tf.sequential()
    model.add(this.tf.layers.dense({
      inputShape: [this.inputShapeNumber],
      activation: "sigmoid",
      units: 5
    }))
    model.add(this.tf.layers.dense({
      inputShape: [5],
      activation: "sigmoid",
      units: this.outputShapeNumber
    }))
    model.add(this.tf.layers.dense({
      activation: "sigmoid",
      units: this.outputShapeNumber
    }))
    model.compile({
      loss: "meanSquaredError",
      optimizer: this.tf.train.adam(.06)
    })
    this.trainModel(input, output, model)
  }

   async trainModel(input, output, model) {
     await model.fit(input, output, {
      epochs: 100,
      callbacks:{
        onEpochEnd: async (epoch, logs)=>{
          console.log(`${epoch+1}. ${logs.loss}`)
        }
      }
    })
      this.testModel(this.test, model);
  }

  testModel(test, model) {
    const result = model.predict(test)
    const finalResult = []
    for (let i = 0; i < Array.from(result.dataSync()).length; i += this.outputShapeNumber) {
      finalResult.push(Array.from(result.dataSync()).slice(i, i + this.outputShapeNumber))
    }
    this.prepareResult(finalResult)
  }

  prepareResult(final) {
    final.forEach(result => {
      console.log(result.map(value=>Math.round(value)))
    })
  }
}

new Classification('TrainSet.txt', 'TestSet.txt')