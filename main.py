#Python Libraries
import pandas as pd
import numpy as np

#Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#Color schemes for data visualization
colors_blue_to_red = ["#1984c5", "#22a7f0", "#63bff0", "#a7d5ed", "#e2e2e2", "#e1a692", "#de6e56", "#e14b31", "#c23728"]

data = pd.read_csv("water_potability.csv")
data.dropna(inplace=True)
X = data.drop(['Potability'], axis=1)
y = data['Potability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

menu_options = {
    1: "Make a water potability prediction",
    2: "Display a histogram of the data",
    3: "Display a scatterplot of the pH vs Hardness and it's relationship to Potability",
    4: "Display a pie chart of the potable and not potable water sample data",
    5: "Display a heatmap of the data",
    0: "Exit program"
}

while True:

    # Display the menu options
    print("**************************************")
    print("=== C964 Computer Science Capstone ===")
    print("         AquaSure Water Analysis      ")
    print("**************************************\n")
    for option, description in menu_options.items():
        print(f"{option}. {description}")

    # Prompt for user choice
    choice = input("Enter your choice (0-5): ")

    # Perform the chosen action
    if choice == "1":
        # Make a water potability prediction
        user_inputs = {}
        fields = {
            'ph': "Enter the value for pH (0 to 14): ",
            'Hardness': "Enter the value for hardness (Capacity of water to precipitate soap in mg/L - typically ranges from 47 to 324): ",
            'Solids': "Enter the value for solids (Total dissolved solids in ppm - typically ranges from 320 to 61228): ",
            'Chloramines': "Enter the value for Chloramines (Amount of Chloramines in ppm - typically ranges from 0 to 14): ",
            'Sulfate': "Enter the value for Sulfate (Amount of Sulfates dissolved in mg/L - typically ranges from 129 to 482): ",
            'Conductivity': "Enter the value for Conductivity (Electrical conductivity of water in µS/cm - typically ranges from 181 to 754): ",
            'Organic_carbon': "Enter the value for organic Carbon (Amount of organic carbon in ppm - typically ranges from 2 to 29): ",
            'Trihalomethanes': "Enter the value for Trihalomethanes (Amount of Trihalomethanes in µg/L - typically ranges from 0 to 124): ",
            'Turbidity': "Enter the value for Turbidity (Measure of light emitting property of water in NTU - typically ranges from 1 to 7): "
        }
        for field, prompt in fields.items():
            while True:
                try:
                    user_inputs[field] = float(input(prompt))
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")

        input_data = pd.DataFrame(user_inputs, index=[0])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            print("\n** The water sample is potable and safe for human consumption. **\n")
        else:
            print("\n** The water is non-potable and NOT safe for human consumption. **\n")

        #Calculate and display the prediction accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Prediction accuracy: {accuracy * 100:.2f}%")

    elif choice == "2":
        # Display a histogram of the data
        data.hist(figsize=(15, 10))
        plt.show()

    elif choice == "3":
        # Display a scatterplot of the data
        potable_color = colors_blue_to_red[3]
        not_potable_color = colors_blue_to_red[5]

        plt.figure(figsize=(10, 8))
        potable_mask = data['Potability'] == 1
        not_potable_mask = data['Potability'] == 0

        plt.scatter(data.loc[potable_mask, 'ph'], data.loc[potable_mask, 'Hardness'], color=potable_color,
                    label='Potable')
        plt.scatter(data.loc[not_potable_mask, 'ph'], data.loc[not_potable_mask, 'Hardness'], color=not_potable_color,
                    label='Not Potable')

        plt.xlabel('pH')
        plt.ylabel('Hardness')
        plt.title('Scatterplot of pH vs Hardness')
        plt.legend()
        plt.show()

    elif choice == "4":
        # Show a pie chart of the potable and not potable water sample data
        potable_counts = data['Potability'].value_counts()
        labels = ['Potable', 'Not Potable']
        colors = [colors_blue_to_red[0], colors_blue_to_red[8]]
        wedgeprops = {'linewidth': 1, 'edgecolor': 'white'}  # Define wedge properties
        plt.pie(potable_counts, labels=labels, autopct='%1.1f%%', colors=colors, wedgeprops=wedgeprops)
        plt.title("Potable vs. Not Potable Water Samples")
        plt.show()

    elif choice == "5":
        # Display a heatmap of the data
        plt.figure(figsize=(15, 8))
        sns.heatmap(data.corr(), cmap=colors_blue_to_red)
        plt.title('Heatmap of Data')
        plt.show()

    elif choice == "0":
        # Exit program
        print("Exiting the program...")
        break

    else:
        # Invalid choice
        print("Invalid choice. Please enter a number between 0 and 5.")
        
