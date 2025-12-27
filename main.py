import json
import os
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch
import undetected_chromedriver as uc
from flask import Flask, request, jsonify
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class Parser():

    def __init__(self, url):

        self.url = url
        self.driver = None

    def create_driver(self):
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass

        options = uc.ChromeOptions()

        options.add_argument('--disable-background-timer-throttling')
        options.add_argument('--disable-backgrounding-occluded-windows')
        options.add_argument('--disable-renderer-backgrounding')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-gpu')

        # параметры для headless
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-features=VizDisplayCompositor')
        options.add_argument('--disable-extensions')
        options.add_argument('--headless=new')

        self.driver = uc.Chrome(
            options=options,
            headless=False,
            use_subprocess=True
        )

        self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        return self.driver


    def parse_comments(self, comment_css = "lq4_29", comment_block = "pdp_ia3"):

        try_to_load_conter = 0

        while (try_to_load_conter != 2):
            try:
                print("try_load, ", try_to_load_conter)

                self.create_driver()
                self.driver.get(self.url)

                wait = WebDriverWait(self.driver, 20)

                block_element = By.CLASS_NAME, comment_block
                comment_element = wait.until(EC.presence_of_element_located(block_element))
                #self.driver.execute_script(f"""window.scrollBy({{top: {4000}, behavior: 'smooth'}});""")

                if (comment_element):

                    self.driver.execute_script("arguments[0].scrollIntoView();", comment_element)

                    element_locator = (By.CLASS_NAME, comment_css)
                    element = wait.until(EC.presence_of_element_located(element_locator))
                    break
            except TimeoutException:
                try_to_load_conter += 1

            if (try_to_load_conter == 3):
                self.driver.quit()
                return None
                # print("Нет комментариев на странице")


        try:
            no_new_comments_count = 0
            comments_counter = 0

            while True:

                comments = self.driver.find_elements(By.CLASS_NAME, comment_css)
                print(f"Текущее количество комментариев: {len(comments)}")

                self.driver.execute_script("arguments[0].scrollIntoView();", comments[-1])
                pause_time = random.uniform(1.5, 2.5)
                time.sleep(pause_time)

                if comments_counter == len(comments):
                    no_new_comments_count += 1
                else:
                    no_new_comments_count = 0
                    comments_counter = len(comments)

                if no_new_comments_count > 10 or len(comments) == 93:
                    break

            all_comments = self.driver.find_elements(By.CLASS_NAME, comment_css)

            print(f"Всего загружено комментариев: {len(all_comments)}")

            comments_texts = []

            for i, review in enumerate(all_comments, 1):
                try:
                    review_text = review.text.strip()
                    if review_text:
                        comments_texts.append(review_text)
                except Exception as e:
                    return None

            comments_df = pd.DataFrame(comments_texts, columns=['text'])
            print(comments_df)

            return comments_df

        except Exception as e:
            return None
        finally:
            self.driver.quit()



class LocalMultiLabelAnalyzer:

    def __init__(self, model_path: str = "./rubert-model", trained_model_path: str = None):

        print(f"загрузка модели из: {model_path}")

        # загрузка токенизатора из папки
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path
        self.max_length = 128

        # инициализия модели
        self.model = None
        self.mlb = None
        self.classes_ = []

        if trained_model_path and os.path.exists(trained_model_path):
            self.load_trained_model(trained_model_path)
        else:
            print("модель не загружена - будет загружена при обучении")

    def prepare_data(self, df: pd.DataFrame, text_column: str, categories_column: str):
        """
        подготовка данных для обучения
        """

        texts = df[text_column].astype(str).tolist()

        all_categories = []
        for item in df[categories_column]:
            if pd.isna(item):
                all_categories.append([])
            elif isinstance(item, str):
                cats = [c.strip().lower() for c in item.split(',')]
                all_categories.append(cats)
            else:
                all_categories.append([])

        # преобразование в бинарный формат
        self.mlb = MultiLabelBinarizer()
        labels = self.mlb.fit_transform(all_categories)
        self.classes_ = self.mlb.classes_

        return texts, labels

    def create_dataset(self, texts, labels):
        """
        Dataset для обучения
        """

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_len,
                    return_tensors='pt'
                )

                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.float)
                }

        return TextDataset(texts, labels, self.tokenizer, self.max_length)

    def train(self, df: pd.DataFrame, text_column: str = "отзыв",
              categories_column: str = "категории",
              save_path: str = "./trained_model",
              epochs: int = 3,
              test_size: float = 0.2,
              resume_from_checkpoint: bool = False):

        print("дообучение модели...")
        print(f"   Размер тестовой выборки: {test_size * 100}%")

        # проверка, есть ли уже обученная модель
        if resume_from_checkpoint and os.path.exists(save_path):
            print(f"Продолжаю обучение из: {save_path}")
            self.load_trained_model(save_path)

        # подготовка данных
        texts, labels = self.prepare_data(df, text_column, categories_column)

        # разделение на train/validation/test
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42
        )

        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.25, random_state=42
        )

        print(f"\n   Разделение данных:")
        print(f"     Train: {len(train_texts)} отзывов ({len(train_texts) / len(texts) * 100:.1f}%)")
        print(f"     Validation: {len(val_texts)} отзывов ({len(val_texts) / len(texts) * 100:.1f}%)")
        print(f"     Test: {len(test_texts)} отзывов ({len(test_texts) / len(texts) * 100:.1f}%)")

        # создание датасетов
        train_dataset = self.create_dataset(train_texts, train_labels)
        val_dataset = self.create_dataset(val_texts, val_labels)
        test_dataset = self.create_dataset(test_texts, test_labels)

        # загрузка модели (если ещё не загружена)
        if self.model is None:
            print(f"\n   Загружаю модель для дообучения...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.classes_),
                problem_type="multi_label_classification"
            )
        else:
            print(f"\n   Использую уже загруженную модель")

        # настройка обучения
        training_args = TrainingArguments(
            output_dir=save_path,
            num_train_epochs=epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f'{save_path}/logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True
        )

        # создание Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        # обучение
        print("\nНачинаю обучение...")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # сохранение модели
        print(f"\nСохраняю дообученную модель в: {save_path}")
        trainer.save_model(save_path)

        # Обновляем self.model на обученную версию
        self.model = trainer.model

        # сохранение информацию о категориях
        joblib.dump(self.mlb, f'{save_path}/mlb.pkl')

        # сохранение списка категорий в файл
        with open(f'{save_path}/categories.txt', 'w', encoding='utf-8') as f:
            for category in self.classes_:
                f.write(f"{category}\n")

        print("Обучение завершено!")

        # оценка на validation данных
        print("\n РЕЗУЛЬТАТЫ НА VALIDATION ДАННЫХ:")
        eval_results = trainer.evaluate()
        for key, value in eval_results.items():
            print(f"   {key}: {value:.4f}")

        # оценка на test данных
        print("\nТЕСТИРОВАНИЕ НА 20% ОТЛОЖЕННЫХ ДАННЫХ:")
        test_predictions = trainer.predict(test_dataset)
        test_pred_probs = torch.sigmoid(torch.tensor(test_predictions.predictions)).numpy()
        test_preds = (test_pred_probs >= 0.4).astype(int)
        print(classification_report(test_labels, test_preds, target_names=self.classes_))

        return trainer, test_texts, test_labels, test_preds

    def compute_metrics(self, eval_pred):
        """
        Вычисление метрик
        """
        predictions, labels = eval_pred

        # преобразуем логиты в вероятности
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.tensor(predictions))

        # бинаризуем с порогом 0.4
        preds = np.zeros(probs.shape)
        preds[probs >= 0.4] = 1

        # вычисляем метрики
        micro_f1 = f1_score(labels, preds, average='micro')
        micro_precision = precision_score(labels, preds, average='micro')
        micro_recall = recall_score(labels, preds, average='micro')

        return {
            'f1': micro_f1,
            'precision': micro_precision,
            'recall': micro_recall
        }

    def load_trained_model(self, model_path: str = "./trained_model"):
        """
        Загрузка дообученной модели
        """
        if not os.path.exists(model_path):
            raise ValueError(f"Папка с моделью не существует: {model_path}")

        print(f"Загружаю дообученную модель из: {model_path}")

        # загружаем модель
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # загружаем MultiLabelBinarizer
        mlb_path = f'{model_path}/mlb.pkl'
        if os.path.exists(mlb_path):
            self.mlb = joblib.load(mlb_path)
            self.classes_ = self.mlb.classes_
        else:
            # пытаемся загрузить категории из файла
            categories_path = f'{model_path}/categories.txt'
            if os.path.exists(categories_path):
                with open(categories_path, 'r', encoding='utf-8') as f:
                    self.classes_ = [line.strip() for line in f if line.strip()]
                print(f"MultiLabelBinarizer не найден, загружены только категории")
            else:
                raise ValueError(f"Не найден файл с категориями: {categories_path}")

        print(f"Модель загружена. Категорий: {len(self.classes_)}")

        return self

    def predict(self, texts: list, threshold: float = 0.4, batch_size: int = 32):
        """
        предсказание категорий для списка текстов
        """

        print(f"Анализирую {len(texts)} текстов...")

        # режим оценки
        self.model.eval()

        all_categories = []

        # обработка батчами для экономии памяти
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Токенизация
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # предсказание
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits

                # преобразование логитов в вероятности
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(logits)

            # преобразуем в категории
            for prob in probs:
                indices = (prob > threshold).nonzero(as_tuple=True)[0]

                if len(indices) > 0:
                    categories = [self.classes_[idx] for idx in indices]

                    # сортируем по вероятности
                    cat_probs = [(prob[idx].item(), self.classes_[idx]) for idx in indices]
                    cat_probs.sort(reverse=True, key=lambda x: x[0])

                    sorted_cats = [cat for _, cat in cat_probs]
                    all_categories.append(sorted_cats)
                else:
                    # если ничего не нашли, берем самую вероятную
                    max_idx = torch.argmax(prob).item()
                    if prob[max_idx] > 0.1:
                        all_categories.append([self.classes_[max_idx]])
                    else:
                        all_categories.append([])

        return all_categories



    def analyze_to_json(self, df: pd.DataFrame,
                        threshold: float = 0.4) -> str:
        """
        анализирует отзывы и возвращает JSON в формате: категория[0] отзыв[1]

        """
        if self.model is None:
            return None

        text_column = df.columns[0]
        texts = df[text_column].astype(str).tolist()

        # предсказываем категории для каждого отзыва
        predicted_categories = self.predict(texts, threshold=threshold)

        # создаем структуру для группировки отзывов по категориям
        category_to_reviews = {}
        other_reviews = []

        # проходим по всем отзывам и их предсказанным категориям
        for i, (text, categories) in enumerate(zip(texts, predicted_categories)):
            if not categories:
                # если категории не найдены, добавляем в "другие"
                other_reviews.append(text)
            else:
                for category in categories:
                    if category not in category_to_reviews:
                        category_to_reviews[category] = []
                    category_to_reviews[category].append(text)

        # если есть отзывы без категорий, добавляем категорию "другие"
        if other_reviews:
            category_to_reviews["другие"] = other_reviews

        # формируем двумерный массив: категория[0] отзыв[1]
        result_array = []

        for category, reviews in category_to_reviews.items():
            for review in reviews:
                result_array.append([category, review])

        # JSON с отступами для читаемости
        result_json = json.dumps(result_array, ensure_ascii=False, indent=2)

        return result_json


    def save_model(self, path: str = "./my_trained_model"):
        """
        сохраняет текущую модель и все необходимые файлы
        """
        if self.model is None:
            raise ValueError("Нет модели для сохранения!")

        print(f"Сохранение модели в: {path}")

        # создание директорию если не существует
        os.makedirs(path, exist_ok=True)

        # сохранение модель
        self.model.save_pretrained(path)

        # сохранение токенизатор
        self.tokenizer.save_pretrained(path)

        # сохранение MultiLabelBinarizer если есть
        if self.mlb is not None:
            joblib.dump(self.mlb, f'{path}/mlb.pkl')

        # сохранение категорий в файл
        with open(f'{path}/categories.txt', 'w', encoding='utf-8') as f:
            for category in self.classes_:
                f.write(f"{category}\n")

        print(f"Модель сохранена в {path}")



app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Парсер комментариев</title>
    </head>
    <body>
        <form action="/category/" method="GET">
            <input type="text" name="url" style="width: 400px;" placeholder="введите url товара озон">
            <button type="submit">Поиск</button>
        </form>
    </body>
    </html>
    '''

@app.route('/category/')
def page():

    page_url = request.args.get('url')

    parser = Parser(page_url)
    comments_df = parser.parse_comments()

    if comments_df is None or comments_df.empty:
        return jsonify({"error": "Комментарии не найдены"}), 404

    config = {
        'rubert_model': "./rubert-model",
        'trained_model': "./моя_обученная_модель",
        'train_data': 'clothes_dataset.csv',
        'test_data': "Тестовый_датасет_2.csv"
    }

    # загрузка/обучение модели
    if os.path.exists(config['trained_model']):
        print("Загружаю обученную модель...")
        analyzer = LocalMultiLabelAnalyzer(
            model_path=config['rubert_model'],
            trained_model_path=config['trained_model']
        )
    else:
        print("Обучаю модель...")
        df_train = pd.read_csv(config['train_data'], encoding='utf-8')
        analyzer = LocalMultiLabelAnalyzer(model_path=config['rubert_model'])
        analyzer.train(
            df=df_train,
            text_column='отзыв',
            categories_column='категории',
            save_path=config['trained_model']
        )


    json_result = analyzer.analyze_to_json(comments_df)
    parsed_json = json.loads(json_result)
    print(parsed_json)


    return json_result, 200, {'Content-Type': 'application/json; charset=utf-8'}



if __name__ == "__main__":

    app.run(debug=True)