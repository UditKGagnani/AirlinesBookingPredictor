import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout='wide')
st.title("Airline's Booking Predictor Project")
st.header("")

st.sidebar.header("Your Airline Preferences")


def user_def_inputs():
    num_passengers = st.sidebar.slider("Number of Passengers", 1, 9, 4)
    sales_channel = st.sidebar.selectbox("Sales Channel", ("Internet", "Mobile"))
    purchase_lead = st.sidebar.slider("Difference between Booking and Travel Date", 1, 730, 365)
    length_of_stay = st.sidebar.slider("Length of Stay", 0, 730, 365)
    flight_hour = st.sidebar.slider("Hour of Departure", 0, 23, 12)
    flight_day = st.sidebar.selectbox("Departure Day", ("Monday", "Tuesday", "Wednesday",
                                                        "Thursday", "Friday", "Saturday", "Sunday"))
    booking_origin = st.sidebar.selectbox("Booking Origin", ('Australia', 'New Zealand', 'India', 'Other',
                                                             'China', 'South Korea', 'Japan', 'Malaysia',
                                                             'Singapore', 'Indonesia', 'Thailand', 'Taiwan'))
    wants_extra_baggage = st.sidebar.selectbox("Wants Extra Baggage", ("Yes", "No"))
    wants_preferred_seat = st.sidebar.selectbox("Wants Preferred Seats", ("Yes", "No"))
    wants_in_flight_meals = st.sidebar.selectbox("Wants In Flight Meals", ("Yes", "No"))
    flight_duration = st.sidebar.slider("Flight Duration", 4.5, 9.5, 7.5)

    data = {"num_passengers": num_passengers,
            "sales_channel": sales_channel,
            "purchase_lead": purchase_lead,
            "length_of_stay": length_of_stay,
            "flight_hour": flight_hour,
            "flight_day": flight_day,
            "booking_origin": booking_origin,
            "wants_extra_baggage": wants_extra_baggage,
            "wants_preferred_seat": wants_preferred_seat,
            "wants_in_flight_meals": wants_in_flight_meals,
            "flight_duration": flight_duration}
    return pd.DataFrame(data, index=[0])


input_df = user_def_inputs()

mapping = {"Yes": 1, "No": 0}

input_df['wants_extra_baggage'] = input_df['wants_extra_baggage'].map(mapping)
input_df['wants_preferred_seat'] = input_df['wants_preferred_seat'].map(mapping)
input_df['wants_in_flight_meals'] = input_df['wants_in_flight_meals'].map(mapping)

mapping = {"Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu",
           "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun"}

input_df['flight_day'] = input_df['flight_day'].map(mapping)

pipe = joblib.load('pipe')
input_df = pipe.transform(input_df)

pca = joblib.load('pca')
input_df = pca.transform(input_df)

abm = joblib.load('airlines_booking_model')
prediction = abm.predict(input_df)

print("Prediction:", str(prediction))

# score = np.round(prediction[0], 2)
# if 90 <= score <= 100:
#     status = "Outstanding"
#     grade = 'O'
# elif 80 <= score < 90:
#     status = "Excellent"
#     grade = 'A+'
# elif 70 <= score < 80:
#     status = "Very Good"
#     grade = 'A'
# elif 60 <= score < 70:
#     status = "Good"
#     grade = 'B+'
# elif 55 <= score < 60:
#     status = "Above Average"
#     grade = 'B'
# elif 50 <= score < 55:
#     status = "Average"
#     grade = 'C'
# elif 45 <= score < 50:
#     status = "Bad"
#     grade = 'C-'
# else:
#     status = "Need a Lot of Improvement"
#     grade = 'D'
#
# score_criteria_df = pd.DataFrame({
#     "Performance Range": ["=90 - <=100", "=80 - <90", "=70 - <80", "=60 - <70", "=55 - <60", "=50 - <55", "=45 - <50",
#                           "44 and Below"],
#     "Grade": ['O', 'A+', 'A', 'B+', 'B', 'C', 'C-', 'D'],
#     "Status": ['Outstanding', 'Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 'Bad',
#                'Need a Lot of Improvement']
# })
#
# scores_column, score_criteria_column = st.columns((1, 1))
#
# with scores_column:
#     st.header("")
#     st.header("")
#     st.subheader(f"\n\nPerformance Index: **{str(score)}**")
#     st.subheader(f"Performance Grade: **{grade}**")
#     st.subheader(f"Performance Status: **{status}**")
#
# with score_criteria_column:
#     st.write("\n\n **Grading System:** ")
#     st.dataframe(score_criteria_df)

st.header("")
st.header("")

if prediction == 1:
    st.subheader(f"\n\n**Yes Booking will be Completed**")
else:
    st.subheader(f"\n\n**No Booking will not be Completed**")
