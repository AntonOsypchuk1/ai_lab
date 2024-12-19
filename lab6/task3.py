# Вхідні дані
data = {
    "Outlook": {"Sunny": {"Yes": 3/10, "No": 2/4}},
    "Humidity": {"Normal": {"Yes": 6/9, "No": 1/5}},
    "Wind": {"Strong": {"Yes": 6/9, "No": 2/5}}
}

# Ймовірності для гри "Yes"
p_yes = (
    data["Outlook"]["Sunny"]["Yes"] *
    data["Humidity"]["Normal"]["Yes"] *
    data["Wind"]["Strong"]["Yes"]
)

# Ймовірності для гри "No"
p_no = (
    data["Outlook"]["Sunny"]["No"] *
    data["Humidity"]["Normal"]["No"] *
    data["Wind"]["Strong"]["No"]
)

# Нормалізація
p_yes_normalized = p_yes / (p_yes + p_no)
p_no_normalized = p_no / (p_yes + p_no)

print(f"Ймовірність гри 'Yes': {p_yes_normalized:.2f}")
print(f"Ймовірність гри 'No': {p_no_normalized:.2f}")

if p_yes_normalized > p_no_normalized:
    print("Прогноз: Гра відбудеться.")
else:
    print("Прогноз: Гра не відбудеться.")
