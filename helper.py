
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
path = './'
variables_df = pd.read_csv(path + 'variables.csv', encoding='ISO-8859-1')
# print(variables_df.head())
variable_dict = dict(zip(variables_df['Variable Name'], variables_df['Renamed_variables']))
# Define variable_dict with appropriate mappings


def hist(merged_data, var_name):
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=merged_data, x=var_name, hue='DIQ010', multiple='stack', palette={1: 'blue', 2: 'orange'})
    plt.xlabel(var_name)
    plt.ylabel('Count')
    plt.title('Histogram of ' + variable_dict[var_name] + ' with Hue' + variable_dict["DIQ010"] ) 
    
    # Ensure var_name column exists and contains valid data
    if var_name in merged_data.columns:
        unique_values = sorted(list(merged_data[var_name].dropna().unique()))
        if len(unique_values) > 2:
            plt.xlim(unique_values[0], unique_values[-3])
        else:
            plt.xlim(unique_values[0], unique_values[-1])
        if len(unique_values) >= 3:
            third_largest_unique_value = unique_values[-3]
        else:
            print("Not enough unique values to find the third largest.")
        
        # Calculate and plot median, mode, and mean
        valid_values = merged_data[var_name][merged_data[var_name] <= third_largest_unique_value]

        median_val = valid_values.median()
        mode_val = valid_values.mode().iloc[0]
        mean_val = valid_values.mean()  
        plt.axvline(median_val, color='r', linestyle='--', label=f'Median: {median_val}')
        plt.axvline(mode_val, color='g', linestyle='-', label=f'Mode: {mode_val}')
        plt.axvline(mean_val, color='b', linestyle='-.', label=f'Mean: {mean_val}')
        
        # Add custom legend for hue values
        handles, labels = plt.gca().get_legend_handles_labels()
        hue_labels = {1: 'Diabetic', 2: 'Non-Diabetic'}
        for hue_value, hue_label in hue_labels.items():
            handles.append(plt.Line2D([0], [0], color=sns.color_palette()[hue_value-1], lw=4))
            labels.append(hue_label)
        
        plt.legend(handles, labels)
    else:
        print(f"{var_name} column not found in the merged data.")
    
    plt.show()
def pie_chart(merged_data,var_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.legend()
    if var_name in merged_data.columns:
        data = merged_data[var_name].value_counts()
        plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
        plt.title(f'Pie Chart of {var_name}')
        plt.show()
    else:
        print(f"{var_name} column not found in the merged data.")

def heatmap(merged_data,var1, var2):
    if var1 in merged_data.columns and var2 in merged_data.columns:
        plt.figure(figsize=(10, 8))
        correlation_matrix = merged_data[[var1, var2]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(f'Heatmap between {var1} and {var2}')
        plt.show()
    else:
        print(f"One or both variables {var1} and {var2} not found in the merged data.")
def save_dict( df ,filename):
    df.rename(columns=variable_dict, inplace=True)
    df.to_csv(filename + '.csv', index=False)
    print(f"File {filename}.csv saved successfully")    

