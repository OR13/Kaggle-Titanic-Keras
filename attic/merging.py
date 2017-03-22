
# def merge_predictions():

#     prediction_files = glob.glob(os.getcwd() + "/pred/*.csv")

#     all_predictions = []

#     passenger_ids = pd.read_csv('./data/test.csv').PassengerId

#     # print(passenger_ids)

#     # all_predictions.append(passenger_ids)

#     for prediction_file in prediction_files:
#         # print(prediction_file)

#         if "all_predictions" in prediction_file:
#             continue

#         prediction = pd.read_csv(prediction_file)

#         name = prediction_file.split('/pred/')[1]
#         # print(name)

#         prediction = prediction.drop('PassengerId', axis=1)
#         prediction = prediction.rename(index=str, columns={"Survived": name})
#         prediction = prediction.astype(int)
#         all_predictions.append(prediction)

#     # print(all_predictions)

#     all_preds = pd.concat(all_predictions, axis=1)
#     all_preds.to_csv('./pred/all_predictions.csv', index=False)


# merge_predictions()


# def combine_predictions():

#     all_predictions = pd.read_csv('./pred/all_predictions.csv')

#     # print(all_predictions.head())

#     def most_common(lst):
#         return max(set(lst), key=lst.count)

#     final_predictions = []

#     for index, row in all_predictions.iterrows():

#         common_prediction = most_common(list(row))
#         # print(common_prediction)
#         final_predictions.append(common_prediction)

#     passenger_ids = pd.read_csv('./data/test.csv').PassengerId

#     df = pd.DataFrame({'Survived': final_predictions})

#     combined = pd.concat([passenger_ids, df], axis=1)

#     combined.to_csv('./data/combined_predictions.csv', index=False)


# combine_predictions()