# Housing forecasting model for real estate agencies (housing_cost)

Real estate agency faced with a problem - realtors spend too much time sorting ads and search for profitable offers. 
Therefore, the speed of their reaction and the quality of analysis are not up to the level of competitors. 
This affects the financial performance of the agency.

The goal is to develop a machine learning model that will help process ads and increase the number of deals and agency profits.

**In Russian:**

Задача следующая: агентство недвижимости столкнулось с проблемой — риелторы тратят слишком много времени на сортировку объявлений и поиск выгодных предложений. Поэтому скорость их реакции и качество анализа не дотягивают до уровня конкурентов. Это сказывается на финансовых показателях агентства.

Цель — разработать модель машинного обучения, которая поможет обрабатывать объявления и увеличит число сделок и прибыль агентства.

# Directory structure

Directory structure inspired by [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science)

```dir
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed (Controled by DVC).
│   ├── processed      <- The final, canonical data sets for modeling (Controled by DVC).
│   └── raw            <- The original, immutable data dump (Controled by DVC).
│
├── docs               <- Contains original project description
│
├── mlartifacts        <- Artifacts (models, signatures etc) keeped by MLFlow (Controled by DVC)
│
├── mlruns             <- Contains runs in MLFlow (Controled by DVC)
│
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
│
├── web             <- Folder with the housing cost prediction server
│
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
 
```
# How to launch prediction service

Выполните команды:

```bash
$ docker pull mvulf/housing_cost_image
$ docker run -it --rm --name=housing_cost_container -p=4000:4000 mvulf/housing_cost_image
```

Затем выполните код [web/test/client.py](https://github.com/mvulf/housing_cost/blob/main/web/test/client.py)

# How to launch modelling

Необходимо установить [poetry](https://python-poetry.org/docs/), если не установлен.

Для инициализации своей системы контроля данных (не обязательно), [установите dvc](https://dvc.org/doc/install)

Далее необходимо склонировать директорию:
```bash
git clone https://github.com/mvulf/housing_cost.git
```

Затем установить требуемые зависимости с помощью [poetry](https://python-poetry.org/docs/cli/), находясь в директории проекта:
```bash
poetry install
```
Эта команда устанавливает зависимости из [pyproject.toml](https://github.com/mvulf/housing_cost/blob/main/pyproject.toml)

В представленном проекты данные синхронизировались с Google-Disk с использованием [DVC](https://dvc.org/) (см файлы в корневой папке `*.dvc`).
Однако доступ к папке ограничен, в связи с чем команда `dvc pull` не выгрузит данные в папку `./data`.

Для проверки работы кода загрузите файл `data.csv` из архива, скаченного по [ссылке](https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view?usp=share_link), по пути: `./data/raw/data.csv`.
Далее запустите код [./src/data/make_datasets.py](https://github.com/mvulf/housing_cost/blob/main/src/data/make_datasets.py) (РЕКОМЕНДУЕТСЯ!). Этот скрипт создаст папки `interim/`, `processed/` в директории `./data/` и создаст там соответствующие датасеты в формате .csv.

Либо, при желании увидеть преобразования шаг за шагом с соответствующими выводами, запустите последовательно [ноутбуки анализа и подготовки данных](https://github.com/mvulf/housing_cost/tree/main/notebooks):
- 1.0-mv-data-understanding.ipynb
- 2.0-mv-eda-baseline.ipynb
- 2.1-mv-eda-preprocessing.ipynb
- 2.2-mv-eda-feature-selection.ipynb

 После чего можно запускать интересующие [ноутбуки моделирования](https://github.com/mvulf/housing_cost/tree/main/notebooks):
 - 3.0-mv-modelling-init.ipynb
 - 3.1-mv-modelling-baseline.ipynb
 - 3.2-mv-modelling-pf-regularization.ipynb
 - 3.3-mv-modelling-random-forest.ipynb
 - 3.4-mv-modelling-grad-boost.ipynb


# Description

**Бизнес-цель**: разработать модель, которая позволила бы агенству недвижимости обойти конкурентов по скорости и качеству совершения сделок.

**Data-Science цель**: разработать регрессионную модель прогнозирования стоимости жилья для агенства недвижимости, взяв за основу имеющийся [датасет](https://drive.google.com/file/d/11-ZNNIdcQ7TbT8Y0nsQ3Q0eiYQP__NIW/view?usp=share_link)
*(датасет также синхронизируется средствами DVC на google drive, но доступ ограничен)*.

На этапе фактического внедрения модели необходимо прибегнуть к A/B-тестированию, сопоставив скорость и качество совершения сделок до внедрения модели и после интеграции модели прогнозирования стоимости жилья.
При этом уже сейчас можно было бы уточнить у заказчика, что он понимает под качеством совершения сделок, какие показатели у агенства сейчас и с какими конкурентами предполагается вести сравнение.

Ввиду отсутсвия на данный момент этой информации, можно предположить, что агенство недвижимости хочет максимизировать среднее отношение стоимости сделки $\text{Value}\_{\text{trans}}$ ко времени совершения этой сделки $t_{\text{trans}}$ (*trans* от англ. *transaction*):

$$\mathbb{E}\left(\frac{\text{Value}\_\text{trans}}{t_\text{trans}}\right) \rightarrow \text{max}$$

Осуществить это можно, максимизируя точность предсказательной модели, так как при предсказании справедливой цены можно значительно ускорить завершение сделки (снизить $t_\text{trans}$), не потеряв, и, возможно, даже увеличив $\text{Value}_\text{trans}$, предотвращая демпинг цены на недвижимость.

Таким образом, это задача *регрессии*, для которой можно ввести следующие метрики качества модели, с учётом наличия истинных значений стоимости (обучение с учителем):
- Средняя абсолютная ошибка, выраженная в процентах (**MAPE**);
- Коэффициент детерминации (**$R^2$**).

Первая метрика позволит оценить, на сколько процентов в среднем предсказание отклоняется от реального значения.
Будем её использовать, так как нет сведений о том, какое значение целевого показателя считать приемлемым.

Вторая метрика покажет, насколько модель лучше, чем если бы все предсказания были бы средним по правильным ответам. 
То есть показывает, какую долю информации о зависимости (дисперсии) смогла уловить модель.

Однако, необходимо прежде оценить распределение целевого признака, может полезно прибегнуть к логарифмическим метрикам.

Описание исходных данных представлено в [1.0-mv-data-understanding.ipynb](https://github.com/mvulf/housing_cost/blob/main/notebooks/1.0-mv-data-understanding.ipynb)

Предварительный анализ данных проведён с использованием `ProfileReport`. Итоговый отчёт можно найти [здесь](https://github.com/mvulf/housing_cost/blob/main/reports/init_profiling.html)

# Tasks

1. Провести разведывательный анализ (EDA) и очистку исходных данных.
Во многих признаках присутствуют дублирующиеся категории, ошибки ввода, жаргонные сокращения и так далее.
Необходимо отыскать закономерности, расшифровать все сокращения, найти синонимы в данных, обработать пропуски и удалить выбросы.

2. Выделить наиболее значимые факторы, влияющие на стоимость недвижимости (Data Preparation)

3. Построить модель для прогнозирования стоимости недвижимости.

4. Разработать веб-сервис, на вход которому поступают данные о некоторой выставленной на продажу недвижимости, а сервис прогнозирует его стоимость.

# Technologies

В этом проекте применяются:
- Система контроля зависимостей [Poetry](https://python-poetry.org/)
- Система контроля данных [DVC](https://dvc.org/)
- Система логирования ML-экспериментов [MLFlow](https://mlflow.org/)
- Контейнеризация [Docker](https://www.docker.com/)

Также применялись библиотеки:
- matplotlib
- seaborn
- pandas
- scipy
- scikit-learn
- ydata-profiling
- flask
- и д.р.

# Results

В ходе работы над проектом были подготовлены train/test датасеты в пропорции 80/20. Последовательность их подготовки представлена в  [ноутбуках анализа и подготовки данных](https://github.com/mvulf/housing_cost/tree/main/notebooks):
- 1.0-mv-data-understanding.ipynb
- 2.0-mv-eda-baseline.ipynb
- 2.1-mv-eda-preprocessing.ipynb
- 2.2-mv-eda-feature-selection.ipynb

Далее, на обучающей выборке применялась 5-ти фолдовая кросс-валидация для оптимизации гиперпараметров, а финальные метрики оценивались на тестовом датасете. Применялись метрики (см обоснование выше):
- Средняя абсолютная ошибка, выраженная в процентах (**MAPE**);
- Коэффициент детерминации (**$R^2$**).

Процедура обучения моделей представлена в [ноутбуках моделирования](https://github.com/mvulf/housing_cost/tree/main/notebooks):
 - 3.0-mv-modelling-init.ipynb
 - 3.1-mv-modelling-baseline.ipynb
 - 3.2-mv-modelling-pf-regularization.ipynb
 - 3.3-mv-modelling-random-forest.ipynb
 - 3.4-mv-modelling-grad-boost.ipynb


Полученные результаты на тестовом датасете представлены в таблице (в порядке их получения):

| Model | R^2 | MAPE [%] |
|---|---|---|
| Linear regression baseline after eda | 0.391 | 53.1 |
| Decision tree baseline after eda | **0.492** | 49.2 |
| Linear regression with StandardTransform | 0.390 | 53.0 |
| Ridge regression with PolyFeatures and StandardTransform | **0.445** | **47.5** |
| Lasso regression with PolyFeatures and StandardTransform | 0.434 | 47.8 |
| ElasticNet regression with PolyFeatures and StandardTransform | 0.431 | 47.8 |
| Init Random Forest | 0.471 | 49.7 |
| Random Forest with manual set parameters | 0.535 | 45.4 |
| **Random Forest with optimized parameters** | **0.610** | **36.8** |
| Gradient Boosting with manual set parameters | 0.534 | 42.8 |
| **Gradient Boosting with optimized parameters** | **0.654** | **34.7** |

Ниже представлены столбчатые диаграммы метрик, сгенерированные средствами MLFlow.
Порядок расположения колонок столбчатых диаграмм соответствует порядку в таблице выше (сверху вниз)

**TEST $R^2$**

![test_r2](https://github.com/mvulf/housing_cost/blob/main/reports/figures/test_r2.png)

**TEST MAPE**

(fraction instead of percentage)

![test_mape](https://github.com/mvulf/housing_cost/blob/main/reports/figures/test_mape.png)

Итого обучено и сохранено средствами MLFlow 11 моделей. Наилучшие показатели на тестовой выборке показал **"Gradient Boosting with optimized parameters"**. Он и был взят для [web-сервиса](https://github.com/mvulf/housing_cost/tree/main/web), предсказывающего стоимость жилья. 

Веб-сервис был собран в виде [контейнера](https://hub.docker.com/r/mvulf/housing_cost_image).
Он принимает на вход датасет, для которого надо сделать предсказание, и возвращает "prediction".
Порядок запуска теста представлен выше.


