// data/questions.ts

export const questionBank: Record<string, { cie: any[], viva: any[] }> = {
  "lab1": {
    cie: [
      { id: 1, question: "Which library is used to load the California Housing dataset?", options: ["matplotlib", "sklearn", "pandas", "numpy"], correct: 1 },
      { id: 2, question: "What does IQR stand for in outlier detection?", options: ["Internal Quality Ratio", "Inter-Quartile Range", "Integer Quantile Rate", "Inner Quartile Region"], correct: 1 },
      { id: 3, question: "Which plot is BEST for visualizing outliers?", options: ["Histogram", "Box Plot", "Scatter Plot", "Pie Chart"], correct: 1 },
      { id: 4, question: "What is the standard threshold for detecting outliers?", options: ["Q3 + 1.5*IQR", "Mean + 2*SD", "Median * 1.5", "Max - Min"], correct: 0 },
      { id: 5, question: "In pandas, which method gives statistical summary?", options: [".info()", ".head()", ".describe()", ".summary()"], correct: 2 }
    ],
    // 👇 VIVA IS NOW MCQ FORMAT
    viva: [
      { id: 1, question: "Why is data preprocessing important?", options: ["To reduce file size", "To clean noise and handle missing values", "To make code faster", "It is not important"], correct: 1 },
      { id: 2, question: "What is the purpose of a Histogram?", options: ["To show correlation", "To show distribution of data", "To show outliers", "To show hierarchy"], correct: 1 },
      { id: 3, question: "What does the 'bins' parameter control?", options: ["Color of bars", "Number of intervals", "Height of plot", "Transparency"], correct: 1 },
      { id: 4, question: "How do you handle missing values in a dataset?", options: ["Delete the file", "Impute with mean/median or drop rows", "Ignore them", "Change column names"], correct: 1 },
      { id: 5, question: "What is a Box Plot mainly used for?", options: ["Showing trends over time", "Showing visualizing spread and outliers", "Comparing categories", "Showing 3D data"], correct: 1 }
    ]
  },
  "lab2": {
    cie: [
      { id: 1, question: "What does the .corr() method compute?", options: ["Covariance", "Correlation Coefficient", "Standard Deviation", "Variance"], correct: 1 },
      { id: 2, question: "Which value represents a strong negative correlation?", options: ["0.9", "0.1", "-0.9", "-0.1"], correct: 2 },
      { id: 3, question: "Which library is used to plot the Heatmap?", options: ["Seaborn", "Pandas", "Numpy", "Scipy"], correct: 0 },
      { id: 4, question: "If two variables have correlation 0, they are...", options: ["Identical", "Strongly related", "Uncorrelated", "Inversely related"], correct: 2 },
      { id: 5, question: "The diagonal of a correlation matrix is always...", options: ["0", "1", "-1", "Depends on data"], correct: 1 }
    ],
    viva: [
      { id: 1, question: "What is Correlation?", options: ["Causation between variables", "Statistical relationship between two variables", "Difference between means", "Sum of errors"], correct: 1 },
      { id: 2, question: "Does correlation imply causation?", options: ["Yes, always", "No", "Only if positive", "Only if negative"], correct: 1 },
      { id: 3, question: "What is a Heatmap?", options: ["A map of temperature", "A graphical representation of data where values are depicted by color", "A line graph", "A scatter plot"], correct: 1 },
      { id: 4, question: "What is the range of Pearson Correlation Coefficient?", options: ["0 to 1", "-1 to 1", "-infinity to infinity", "0 to 100"], correct: 1 },
      { id: 5, question: "Which library is best for statistical data visualization?", options: ["Math", "Seaborn", "Sys", "OS"], correct: 1 }
    ]
  },
  // Add more labs (3-10) with the same structure
};