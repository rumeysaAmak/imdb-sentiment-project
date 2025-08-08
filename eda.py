from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = load_dataset("imdb")

print("Veri Seti Boyutları:")
for split in dataset:
    print(f"{split}: {len(dataset[split])} örnek")

print("\nİlk 3 Train Örneği:")
for i in range(3):
    print(f"Label: {dataset['train'][i]['label']} - Text: {dataset['train'][i]['text'][:100]}...")

train_labels = [ex["label"] for ex in dataset["train"]]
label_counts = pd.Series(train_labels).value_counts()
label_map = {0: "NEGATIVE", 1: "POSITIVE"}
print("\nSınıf Dağılımı (Train):")
print(label_counts.rename(index=label_map))

train_texts = dataset["train"]["text"]
test_texts = dataset["test"]["text"]

lengths_words_train = [len(t.split()) for t in train_texts]
lengths_words_test = [len(t.split()) for t in test_texts]

print("\nYorum Uzunluğu İstatistikleri (Train):")
print(f"Ortalama kelime: {np.mean(lengths_words_train):.2f}, Medyan: {np.median(lengths_words_train)}, Max: {np.max(lengths_words_train)}")
print("Yorum Uzunluğu İstatistikleri (Test):")
print(f"Ortalama kelime: {np.mean(lengths_words_test):.2f}, Medyan: {np.median(lengths_words_test)}, Max: {np.max(lengths_words_test)}")

df_train = pd.DataFrame({"text": train_texts, "label": train_labels, "length": lengths_words_train})
avg_len_per_class = df_train.groupby("label")["length"].mean().rename(index=label_map)
print("\nSınıf Bazında Ortalama Uzunluk (Train):")
print(avg_len_per_class)

print("\nEn kısa 3 yorum:")
shortest = df_train.nsmallest(3, "length")
for _, row in shortest.iterrows():
    print(f"({label_map[row['label']]}) {row['text'][:100]}...")

print("\nEn uzun 3 yorum:")
longest = df_train.nlargest(3, "length")
for _, row in longest.iterrows():
    print(f"({label_map[row['label']]}) {row['text'][:100]}...")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.hist(lengths_words_train, bins=50, color="skyblue", edgecolor="black")
plt.title("Yorum Uzunluğu Dağılımı (Kelime)")
plt.xlabel("Kelime Sayısı")
plt.ylabel("Frekans")

plt.subplot(1,2,2)
plt.boxplot(lengths_words_train, vert=False, patch_artist=True)
plt.title("Yorum Uzunluğu Kutu Grafiği")
plt.xlabel("Kelime Sayısı")
plt.tight_layout()
plt.show()
