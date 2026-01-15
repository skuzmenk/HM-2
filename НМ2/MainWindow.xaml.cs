using System;
using System.Text;
using System.Windows;
using System.Windows.Controls;

namespace НМ2
{
    public partial class MainWindow : Window
    {
        //test
        private NeuralNetwork nn;
        public MainWindow()
        {
            InitializeComponent();
            nn = new NeuralNetwork(4, 3, 0.05); 
        }
        private async void TrainButton_Click(object sender, RoutedEventArgs e)
        {
            OutputText.Text = "Навчання...";
            await System.Threading.Tasks.Task.Run(() => TrainNetwork());
            OutputText.Text += "\nНавчання завершено.";
        }
        private void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            double[] x = new double[4];
            x[0] = double.Parse(Input1.Text);
            x[1] = double.Parse(Input2.Text);
            x[2] = double.Parse(Input3.Text);
            x[3] = double.Parse(Input4.Text);
            double[] probs = nn.Predict(x);
            string[] classes = { "Setosa", "Versicolor", "Virginica" };
            int predicted = 0;
            double maxProb = probs[0];
            for (int i = 1; i < probs.Length; i++)
            {
                if (probs[i] > maxProb)
                {
                    maxProb = probs[i];
                    predicted = i;
                }
            }
                OutputText.Text = "Ймовірності:\n" +
                                  "Setosa: " + probs[0].ToString("F2") + "\n" +
                                  "Versicolor: " + probs[1].ToString("F2") + "\n" +
                                  "Virginica: " + probs[2].ToString("F2") + "\n\n" +
                                  "Передбачено: " + classes[predicted];
        }
        private void TrainNetwork()
        {
            double[][] X =
            {
                new double[] {5.1, 3.5, 1.4, 0.2},
                new double[] {7.0, 3.2, 4.7, 1.4}, 
                new double[] {6.3, 3.3, 6.0, 2.5} 
            };
            int[] y = { 0, 1, 2 };
            for (int epoch = 0; epoch < 500; epoch++)
            {
                for (int i = 0; i < X.Length; i++)
                {
                    nn.Train(X[i], y[i]);
                }
            }
        }
    }
    public class NeuralNetwork
    {
        private int inputSize;
        private int outputSize;
        private double[,] weights;
        private double learningRate;
        private Random rand = new Random();
        public NeuralNetwork(int inputSize, int outputSize, double learningRate)
        {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
            this.learningRate = learningRate;
            weights = new double[outputSize, inputSize];
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    weights[i, j] = rand.NextDouble() * 0.2 - 0.1;
                }
            }
        }

        public double[] Predict(double[] inputs)
        {
            double[] z = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                for (int j = 0; j < inputSize; j++)
                {
                    z[i] += weights[i, j] * inputs[j];
                }
            }
            return Softmax(z);
        }
        public void Train(double[] inputs, int targetClass)
        {
            double[] yPred = Predict(inputs);
            double[] target = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            { 
                target[i] = (i == targetClass) ? 1.0 : 0.0; 
            }

            for (int i = 0; i < outputSize; i++)
            {
                double error = target[i] - yPred[i];
                for (int j = 0; j < inputSize; j++)
                {
                    weights[i, j] += learningRate * error * inputs[j];
                }
            }
        }

        private double[] Softmax(double[] z)
        {
            double max = z[0];
            for (int i = 1; i < z.Length; i++)
            {
                if (z[i] > max)
                    max = z[i];
            }
            double[] exp = new double[z.Length];
            double sum = 0.0;
            for (int i = 0; i < z.Length; i++)
            {
                exp[i] = Math.Exp(z[i] - max);
                sum += exp[i];
            }
            double[] result = new double[z.Length];
            for (int i = 0; i < z.Length; i++)
            {
                result[i] = exp[i] / sum;
            }
            return result;
        }
    }
}
