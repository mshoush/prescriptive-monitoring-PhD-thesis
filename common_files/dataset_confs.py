import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
treatment_col = {}
time_to_event = {}
pos_treatment = {}
neg_treatment = {}

pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}


logs_dir = f"./../prepared_data/"
# /home/mshoush/Desktop/uncertainity/uncer_2/CatBoost_uncer/data"
#logs_dir = "~/phd/code/Mahmoud_PrescriptiveProcessMonitoring/prepared_data"
#### BPIC2017 settings ####
#TODO change file name for pickle
bpic2017_dict = {"bpic2017": "bpic2017/data_with_inter_case_features_bpic2017.parquet"}
#bpic2012_dict = {"bpic2012": "bpic2012/prepared_treatment_outcome_time_to_event_bpic2012.csv"}


for dataset, fname in bpic2017_dict.items():
    filename[dataset] = os.path.join(logs_dir, fname)
    # min cols
    # standardize column names: ['case_id', 'activity', 'resource', 'timestamp'] for all logs
    case_id_col[dataset] = 'case_id'
    activity_col[dataset] = 'activity'
    resource_col[dataset] = 'resource'
    timestamp_col[dataset] = 'timestamp'

    # label/outcome col
    label_col[dataset] = "label"
    pos_label[dataset] = "regular"  # negative outcome that we don't need to predict
    neg_label[dataset] = "deviant"  # positive outcome that will be predicted
    # treatment col
    treatment_col[dataset] = 'Treatment1'
    pos_treatment[dataset] = "Treatment"  # do treatment
    neg_treatment[dataset] = "Control"  # do not treat

    time_to_event[dataset] = 'remtime' # for regression
    
    # features for classifier
    dynamic_cat_cols[dataset] = ['activity', 'resource', 'lifecycle:transition']#["Activity", 'org:resource', 'Action', 'EventOrigin', 'lifecycle:transition', "Accepted", "Selected", 'time_to_event_m']
    static_cat_cols[dataset] = ['applicationtype', 'loangoal', 'action', 'eventorigin', 'accepted', 'selected']#['ApplicationType', 'LoanGoal']  #static attributes, no need for predicting in suffix predictions
    dynamic_num_cols[dataset] = ['firstwithdrawalamount', 'monthlycost', 'numberofterms', 'offeredamount', 'creditscore', 'event_nr', 'month', 'weekday', 'hour', 'open_cases', 'hour_of_day', 'day_of_week', 'day_of_month', 'month_of_year', 'time_to_last_event_days', 'nr_ongoing_cases', 'interval', 'nr_past_events', 'arrival_rate', 'case_creation_rate', 'case_completion_rate', 'elapsed']#['FirstWithdrawalAmount', 'MonthlyCost', 'NumberOfTerms', 'OfferedAmount', 'CreditScore', "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour","open_cases"]
    static_num_cols[dataset] = ['requestedamount', 'case_length', 'Mail_and_Online_Count', 'Online_Only_Count', 'Total_Offers', 'event']#['NumberOfOffers' ,'RequestedAmount',] #static attributes, no need for predicting in suffix predictions
    


#bpic2012_dict = {"bpic2012": "bpic2012/prepared_treatment_outcome_bpic2012.csv"}
#bpic2012_dict = {"bpic2012": "bpic2012/prepared_treatment_outcome_time_to_event_bpic2012.csv"}
bpic2012_dict = {"bpic2012": "bpic2012/data_with_inter_case_features_bpic2012.parquet"}


#BPI012
for dataset, fname in bpic2012_dict.items():
    #logs_dir = "./data/" + dataset + "/"
    filename[dataset] = os.path.join(logs_dir, fname)
    #print(f"filename: {filename}")
    # min cols
    # standardize column names: ['case_id', 'activity', 'resource', 'timestamp'] for all logs
    case_id_col[dataset] = 'case_id'
    activity_col[dataset] = 'activity'
    resource_col[dataset] = 'resource'
    timestamp_col[dataset] = 'timestamp'

    # label/outcome col
    label_col[dataset] = "label"
    pos_label[dataset] = "regular"  # negative outcome that we don't need to predict
    neg_label[dataset] = "deviant"  # positive outcome that will be predicted
    # treatment col
    treatment_col[dataset] = 'Treatment1'
    pos_treatment[dataset] = "Treatment"  # do treatment
    neg_treatment[dataset] = "Control"  # do not treat

    time_to_event[dataset] = 'remtime' # for regression

    # features for classifier
    dynamic_cat_cols[dataset] = ['activity', 'resource']#["Activity", 'Resource', 'time_to_event_m'] #'Action', 'EventOrigin', 'lifecycle:transition', "Accepted", "Selected"]
    static_cat_cols[dataset] = []#['event']  #static attributes, no need for predicting in suffix predictions
    dynamic_num_cols[dataset] = ['event_nr', 'hour_of_day', 'day_of_week', 'day_of_month', 'month_of_year', 'time_to_last_event_days', 'nr_ongoing_cases', 'interval', 'nr_past_events', 'arrival_rate', 'case_creation_rate', 'case_completion_rate', 'elapsed',]#["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour","open_cases",]
    static_num_cols[dataset] = ['amount_req', 'case_length', 'Total_Offers', 'event']#['NumberOfOffers', "AMOUNT_REQ",] #static attributes, no need for predicting in suffix predictions
    
    





# #bpic2012_dict = {"bpic2012": "bpic2012/prepared_treatment_outcome_bpic2012.csv"}
# traficFines_dict = {"trafficFines": "trafficFines/prepared_treatment_outcome_time_to_event_trafficFines.csv"}


# #BPI012
# for dataset, fname in traficFines_dict.items():
#     #logs_dir = "./data/" + dataset + "/"
#     filename[dataset] = os.path.join(logs_dir, fname)
#     #print(f"filename: {filename}")
#     # min cols
#     case_id_col[dataset] = "Case ID"
#     activity_col[dataset] = "Activity"
#     #resource_col[dataset] = 'Resource'
#     timestamp_col[dataset] = 'time:timestamp'
#     # label/outcome col
#     label_col[dataset] = "label"
#     neg_label[dataset] = "regular"  # negative outcome that we don't need to predict
#     pos_label[dataset] = "deviant"  # positive outcome that will be predicted
#     # treatment col
#     treatment_col[dataset] = 'treatment'
#     pos_treatment[dataset] = "treat"  # do treatment
#     neg_treatment[dataset] = "noTreat"  # do not treat
#     time_to_event[dataset] = 'time_to_event_m'
#     # features for classifier
#     dynamic_cat_cols[dataset] = ["Activity",'time_to_event_m'] #'Action', 'EventOrigin', 'lifecycle:transition', "Accepted", "Selected"]
#     static_cat_cols[dataset] = ['event']  #static attributes, no need for predicting in suffix predictions
#     dynamic_num_cols[dataset] = ["timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour","open_cases",]
#     static_num_cols[dataset] = [] #static attributes, no need for predicting in suffix predictions

