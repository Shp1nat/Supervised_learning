#include <iostream>
#include <random>
#include <vector>
using namespace std;
random_device rd; //random
mt19937 gen(rd()); //random

double sigmoid(double x) { //функция активации sigmoid f(x) = 1 / (1 + e ^ (-x))
    return 1 / (1 + exp(-x));
}

double deriv_sigmoid(double x) { // Производная от sigmoid f'(x) = f(x) * (1 - f(x))
    double fx = sigmoid(x);
    return fx * (1 - fx);
}

double mse_loss(vector<double> trueResults, vector<double> predicatedResults) { //функция вычисления среднеквадратической ошибки потери (меньшие потери = лучшие предсказания)
    //trueResult.size() = predicatedResult.size()
    double sum = 0.0;
    for (int i = 0; i < trueResults.size(); i++) {
        sum += pow(trueResults[i] - predicatedResults[i], 2); //сумма квадрата разниц истины и предсказания
    }
    return sum / trueResults.size(); //делим сумму на кол-во элементов
}

class NeuralNetwork {
    double w1, w2, w3, w4, w5, w6; //веса для h1, h2, o1
    double b1, b2, b3; //смещения для h1, h2, o1
public:
    NeuralNetwork() {
        uniform_int_distribution<> makeRandom(0, RAND_MAX);
        //инициализация весов
        w1 = makeRandom(gen) / double(RAND_MAX); //рандомное число от 0 до 1
        w2 = makeRandom(gen) / double(RAND_MAX);
        w3 = makeRandom(gen) / double(RAND_MAX);
        w4 = makeRandom(gen) / double(RAND_MAX);
        w5 = makeRandom(gen) / double(RAND_MAX);
        w6 = makeRandom(gen) / double(RAND_MAX);
        //инициализация смещений
        b1 = makeRandom(gen) / double(RAND_MAX); //рандомное число от 0 до 1
        b2 = makeRandom(gen) / double(RAND_MAX);
        b3 = makeRandom(gen) / double(RAND_MAX);
    }
    double feedForward(vector<double> x) { //прогоняем наши входные данные через нейроны сети (скрытого и выводного слоев) и получаем результат
        //x.size() = 2
        double h1 = sigmoid(w1 * x[0] + w2 * x[1] + b1); //вычисляем h1 скрытого слоя
        double h2 = sigmoid(w3 * x[0] + w4 * x[1] + b2); //вычисляем h2 скрытого слоя
        double o1 = sigmoid(w5 * h1 + w6 * h2 + b3); //вычисляем o1 выводного слоя
        return o1; //возвращаем ответ
    }
    void train(vector<vector<double>> data, vector<double> trueResults) { //тренировка нейтронной сети = стремление к минимизации ее потерь
        double learningRating = 0.1;
        int iterations = 1000; // количество циклов во всём наборе данных
        for (int interation = 0; interation < iterations; interation++) {
            for (int i = 0; i < data.size(); i++) {
                vector<double> x = data[i]; //получаем текущий элемент
                double trueResult = trueResults[i]; //получаем истиный результат для текущего элемента
                double sum_h1 = w1 * x[0] + w2 * x[1] + b1; //получаем сумму для подстановки в сигмоиду для нейрона h1
                double h1 = sigmoid(sum_h1); //получаем h1 из сигмоиды, куда передали сумму для нейрона h2
                double sum_h2 = w3 * x[0] + w4 * x[1] + b2; //получаем сумму для подстановки в сигмоиду для нейрона h2
                double h2 = sigmoid(sum_h2); //получаем h2 из сигмоиды, куда передали сумму для нейрона h2
                double sum_o1 = w5 * h1 + w6 * h2 + b3; //получаем сумму для подстановки в сигмоиду для нейрона o1
                double o1 = sigmoid(sum_o1); //получаем o1 из сигмоиды, куда передали сумму для нейрона o1
                double predicatedResult = o1; //результат в o1 и есть предсказанный нейронной сетью результат
                double d_L_d_predicatedResult = -2 * (trueResult - predicatedResult); //d(L) / d(predicatedResult) - выведенная формула производной квадратичной функции потерь по отношению к предсказанному результату
                //L(w1, w2, w3, w4, w5, w6, b1, b2, b3) - многовариантная функция потери
                //нейрон o1
                double d_predicatedResult_d_w5 = h1 * deriv_sigmoid(sum_o1); //d(ypred) / d(w5) 
                double d_predicatedResult_d_w6 = h2 * deriv_sigmoid(sum_o1); //d(ypred) / d(w6)
                double d_predicatedResult_d_b3 = deriv_sigmoid(sum_o1); //d(ypred) / d(b3)
                double d_predicatedResult_d_h1 = w5 * deriv_sigmoid(sum_o1); //d(ypred) / d(h1)
                double d_predicatedResult_d_h2 = w6 * deriv_sigmoid(sum_o1); //d(ypred) / d(h2) 
                //нейрон h1
                double d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1); //d(h1) / d(w1)
                double d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1); //d(h2) / d(w2)
                double d_h1_d_b1 = deriv_sigmoid(sum_h1);  //d(h1) / d(b1)
                //нейрон h2
                double d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2); //d(h2) / d(w3)
                double d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2); //d(h2) / d(w4) 
                double d_h2_d_b2 = deriv_sigmoid(sum_h2); //d(h2) / d(b2)
                //обновляем нейрон h1
                w1 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_h1 * d_h1_d_w1;
                w2 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_h1 * d_h1_d_w2;
                b1 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_h1 * d_h1_d_b1;
                //обновляем нейрон h2
                w3 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_h2 * d_h2_d_w3;
                w4 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_h2 * d_h2_d_w4;
                b2 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_h2 * d_h2_d_b2;
                //обновляем нейрон o1
                w5 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_w5;
                w6 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_w6;
                b3 -= learningRating * d_L_d_predicatedResult * d_predicatedResult_d_b3;
            }
            if (interation % 50 == 0) { //каждые 50 итераций выводим результат наших потерь из функции MSE
                vector<double> predicatedResults;
                for (int i = 0; i < data.size(); i++) {
                    predicatedResults.push_back(feedForward(data[i]));
                }
                cout << "Interation: " << interation << ", MSE: " << mse_loss(trueResults, predicatedResults) << endl;
            }
        }
    }
};

struct Fruit { //структура фрукта, имеющего вес и диаметр
    Fruit() { //конструктор без параметров
        weight = 0;
        diameter = 0;
    }
    Fruit(double w, double d) { //конструктор с параметрами
        weight = w;
        diameter = d;
    }
    double weight; //вес
    double diameter; //диаметр
};

bool isInputRight(int choice, double weight, double diameter) { //проверка на ограничения
    if (choice == 0 && weight >= 2.0 && weight <= 4.0 && diameter >= 15 && diameter <= 20) {
        return true;
    }
    if (choice == 1 && weight >= 4.0 && weight <= 10.0 && diameter >= 20 && diameter <= 30) {
        return true;
    }
    return false;
}

int main() {
        //массив фруктов для тренировки нейронной сети
        //арбуз - 1
        //дыня - 0
        //арбуз: средний вес: 4-10 кг, средний диаметр: 20-30 см
        //дыня: средний вес: 2-4 кг, средний диаметр: 15-20 см
        //смещение по весу -4.0
        //смещение по диаметру -20.0
    int count = 0, choice = 0; //объявляем + инициализируем переменные для ввода кол-ва фруктов и типа фрукта (0 - дыня, 1 - арбуз)
    cout << "Watermelon(1) weight range: [4, 10];" << endl << "Watermelon(1) diameter range: [20, 30];" << endl //описание ввода
        << "Melon(0) weight range: [2, 4];" << endl << "Melon(0) diameter range: [15, 20]" << endl //описание ввода
        << "Enter the number of fruits : "; //описание ввода
    cin >> count; //получаем кол-во фруктов, которые введем для тренировки нейтронной сети
    vector<Fruit> fruits; //создаем массив данных, которые введем для фруктов
    vector<double> trueResults; //создаем массив для точных результатов
    fruits.reserve(count); //выделяем место под введенное кол-во фруктов
    trueResults.reserve(count); //выделяем место под введенное кол-во фруктов
    double weight = 0.0, diameter = 0.0; //объявляем + инициализируем переменные для ввода веса, диаметра
    for (int i = 0; i < count; ++i) { //для всего кол-ва фруктов
        cout << "Enter type of fruit (0/1), its weight and diameter: "; //просим ввести данные о фрукте
        cin >> choice >> weight >> diameter; //получаем тип фрукта, вес и диаметр
        if (isInputRight(choice, weight, diameter)) { //проверяем на верность ввода
            if (choice == 0) { //если введенный фрукт - дыня,
                trueResults.push_back(0.0); //то добавляем данные об этом в массив точных результатов
            }
            else {
                trueResults.push_back(1.0); //иначе, добавляем данные о том, что это арбуз
            }
            fruits.push_back({ weight - 4.0, diameter - 20.0 }); //добавляем данные о фрукте в массив фруктов
        }
        else { //если пользователь вышел за ограничения вввода, выводим сообщение об ошибке, не засчитываем эту итерацию
            cout << "Input error, try again" << endl;
            --i;
        }
    }

    NeuralNetwork network; //создаем нейронную сеть
    vector<vector<double>> data; //объявляем массив чистых данных о каждом экземпляре
    data.reserve(fruits.size()); //выделяем память на кол-во фруктов
    for (int i = 0; i < fruits.size(); ++i) { //для всего кол-ва фруктов
        data.push_back({ fruits[i].weight, fruits[i].diameter }); //добавляем чистые данные каждого фрукта в массив 
        //(структура Fruit является посредником между вводом данных и добавлением их в массив чистых данных для сети)
    }
    network.train(data, trueResults); //вызываем метод тренировки нейронной сети, куда передаем наш массив чистых данных о фруктах, а также массив точных результатах (о том, кем является фрукт)

    while (true) { //бесконечный цикл
        cout << "Enter weight and diameter of fruit: "; //просим ввести данные о фрукте
        cin >> weight >> diameter; //получаем данные 
        cout << "The results with weight " << weight << " and diameter " << diameter << " = " << network.feedForward({ weight - 4.0, diameter - 20.0 }) << endl; //выводим результат сети по введенным данным
    }

    return 0;
}