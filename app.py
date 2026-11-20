from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# CSV ko data/ folder ya project ke root folder mein dhoondhega
PROJECT_FOLDER = Path(__file__).parent
CSV_NAME = "uber_trips_dataset_50k.csv"
DATA_FILE = PROJECT_FOLDER / "data" / CSV_NAME

if not DATA_FILE.exists():
    DATA_FILE = PROJECT_FOLDER / CSV_NAME


st.set_page_config(
    page_title="Uber Trips Dashboard",
    page_icon="🚕",
    layout="wide"
)


@st.cache_data
def load_data(csv_path):
    data = pd.read_csv(csv_path)

    data["pickup_time"] = pd.to_datetime(
        data["pickup_time"], errors="coerce"
    )
    data["drop_time"] = pd.to_datetime(
        data["drop_time"], errors="coerce"
    )

    data["trip_duration_min"] = (
        data["drop_time"] - data["pickup_time"]
    ).dt.total_seconds() / 60

    data["pickup_hour"] = data["pickup_time"].dt.hour
    data["pickup_day"] = data["pickup_time"].dt.day_name()
    data["pickup_date"] = data["pickup_time"].dt.date

    return data


if not DATA_FILE.exists():
    st.error(
        "CSV file nahi mili. CSV ko app.py ke paas data folder mein "
        "rakhein, ya project folder ke root mein."
    )
    st.stop()

df = load_data(DATA_FILE)


# Dashboard heading
st.title("🚕 Uber Trips Operations Dashboard")
st.caption(
    "Trips, cancellations, fares aur payments ko city aur time ke hisaab se dekhein."
)


# Sidebar filters
with st.sidebar:
    st.header("Filters")
    available_dates = sorted(df["pickup_date"].dropna().unique())

    date_range = st.date_input(
        "Pickup date range",
        value=(available_dates[0], available_dates[-1]),
        min_value=available_dates[0],
        max_value=available_dates[-1]
    )

    cities = st.multiselect(
        "City",
        options=sorted(df["city"].dropna().unique()),
        default=sorted(df["city"].dropna().unique())
    )

    statuses = st.multiselect(
        "Trip status",
        options=sorted(df["status"].dropna().unique()),
        default=sorted(df["status"].dropna().unique())
    )

    payment_methods = st.multiselect(
        "Payment method",
        options=sorted(df["payment_method"].dropna().unique()),
        default=sorted(df["payment_method"].dropna().unique())
    )

    page = st.radio(
        "Dashboard section",
        ["Overview", "Insights", "Data Quality"]
    )


# Selected date range samjhein
if isinstance(date_range, (tuple, list)):
    start_date, end_date = date_range
else:
    start_date = end_date = date_range


# Saare filters apply karein
filtered = df[
    df["pickup_date"].between(start_date, end_date)
    & df["city"].isin(cities)
    & df["status"].isin(statuses)
    & df["payment_method"].isin(payment_methods)
].copy()


if page == "Data Quality":
    st.header("Data Quality")
    st.caption("Ye checks poore dataset par hain; sidebar filters inhe nahi badalte.")

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    zero_distance = int((df["distance_km"] <= 0).sum())
    zero_duration = int((df["trip_duration_min"] <= 0).sum())
    negative_duration = int((df["trip_duration_min"] < 0).sum())

    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Missing cells", f"{missing_cells:,}")
    q2.metric("Duplicate rows", f"{duplicate_rows:,}")
    q3.metric("Zero-distance trips", f"{zero_distance:,}")
    q4.metric("Zero-duration trips", f"{zero_duration:,}")
    q5.metric("Negative-duration trips", f"{negative_duration:,}")

    st.subheader("Pickup coordinates by city")
    city_coordinates = (
        df.groupby("city")[["pickup_lat", "pickup_lng"]]
        .median()
        .round(2)
    )
    st.dataframe(city_coordinates, use_container_width=True)

    # Pickup aur drop ke beech straight-line distance ka estimate
    pickup_lat = np.radians(df["pickup_lat"].to_numpy())
    drop_lat = np.radians(df["drop_lat"].to_numpy())
    delta_lat = drop_lat - pickup_lat
    delta_lng = np.radians(
        df["drop_lng"].to_numpy() - df["pickup_lng"].to_numpy()
    )

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(pickup_lat)
        * np.cos(drop_lat)
        * np.sin(delta_lng / 2) ** 2
    )
    straight_line_km = 6371 * 2 * np.arcsin(np.sqrt(a))

    valid_distance = df["distance_km"] > 0
    ratio = (
        straight_line_km[valid_distance.to_numpy()]
        / df.loc[valid_distance, "distance_km"].to_numpy()
    )

    left, right = st.columns(2)
    left.metric(
        "Median straight-line distance",
        f"{np.median(straight_line_km):.2f} km"
    )
    right.metric(
        "Median reported trip distance",
        f"{df['distance_km'].median():.2f} km"
    )

    st.write(
        "Straight-line distance reported distance se zyada hai:",
        f"{(ratio > 1).mean() * 100:.1f}% rows mein."
    )

    st.warning(
        "Har city ke pickup-coordinate medians lagbhag 39°N, 97°W ke paas hain. "
        "Ye coordinates city labels se match nahi karte. Map ya location analysis "
        "se pehle coordinates verify karein."
    )

    st.info(
        "Zero-distance aur zero-duration rows ko trip counts mein rakhein, "
        "lekin duration averages ya per-kilometre calculations se exclude karein."
    )


if filtered.empty:
    st.warning(
        "In filters ke liye koi trip nahi mili. Date range ya filters badal kar dekhein."
    )


if page == "Overview":
    completed = filtered[filtered["status"] == "Completed"]

    # KPI cards
    st.subheader("Selected trips ke KPIs")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("Total trips", f"{len(filtered):,}")

    completion_rate = (filtered["status"] == "Completed").mean() * 100
    k2.metric("Completion rate", f"{completion_rate:.1f}%")

    cancellation_rate = (filtered["status"] == "Cancelled").mean() * 100
    k3.metric("Cancellation rate", f"{cancellation_rate:.1f}%")

    k4.metric(
        "Average fare",
        f"{filtered['fare_amount'].mean():.2f}",
        help="Source file mein fare ki currency specify nahi hai."
    )

    k5.metric(
        "Average distance",
        f"{filtered['distance_km'].mean():.2f} km"
    )

    k6.metric(
        "Completed fare total",
        f"{completed['fare_amount'].sum():,.2f}",
        help="Sirf Completed trips ka fare; currency source mein specify nahi hai."
    )

    # Trips by date
    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Trips by pickup date")

        daily_trips = (
            filtered.groupby("pickup_date")
            .size()
            .rename_axis("Pickup date")
            .reset_index(name="Trips")
        )
        daily_trips["Pickup date"] = pd.to_datetime(daily_trips["Pickup date"])

        st.line_chart(
            daily_trips,
            x="Pickup date",
            y="Trips",
            x_label="Pickup date",
            y_label="Number of trips"
        )

        busiest_day = daily_trips.loc[daily_trips["Trips"].idxmax()]
        st.caption(
            f"Sabse zyada trips {busiest_day['Pickup date'].date()} ko hui: "
            f"{busiest_day['Trips']}."
        )

    # Trip status mix
    with right:
        st.subheader("Trips by status")

        status_counts = (
            filtered["status"]
            .value_counts()
            .rename_axis("Status")
            .reset_index(name="Trips")
        )

        st.bar_chart(
            status_counts,
            x="Status",
            y="Trips",
            x_label="Trip status",
            y_label="Number of trips"
        )

        st.caption(
            "Completed, Cancelled aur No-Show trips ka comparison."
        )

    # City chart
    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Completed trips by city")

        if completed.empty:
            st.info("Completed trips dekhne ke liye status filter mein Completed chunein.")
        else:
            completed_by_city = (
                completed.groupby("city")
                .size()
                .rename_axis("City")
                .reset_index(name="Completed trips")
            )

            st.bar_chart(
                completed_by_city,
                x="City",
                y="Completed trips",
                x_label="City",
                y_label="Completed trips"
            )

            top_city = completed_by_city.loc[
                completed_by_city["Completed trips"].idxmax()
            ]
            st.caption(
                f"{top_city['City']} mein sabse zyada completed trips: "
                f"{top_city['Completed trips']}."
            )

    # Payment method chart
    with right:
        st.subheader("Trips by payment method")

        payment_counts = (
            filtered["payment_method"]
            .value_counts()
            .rename_axis("Payment method")
            .reset_index(name="Trips")
        )

        st.bar_chart(
            payment_counts,
            x="Payment method",
            y="Trips",
            x_label="Payment method",
            y_label="Number of trips"
        )

        most_used = payment_counts.iloc[0]
        st.caption(
            f"{most_used['Payment method']} sabse zyada use hua: "
            f"{most_used['Trips']} trips."
        )

    # Hourly volume and weekday cancellations
    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Trips by pickup hour")

        hourly_trips = (
            filtered.groupby("pickup_hour")
            .size()
            .reindex(range(24), fill_value=0)
            .rename_axis("Pickup hour")
            .reset_index(name="Trips")
        )

        st.bar_chart(
            hourly_trips,
            x="Pickup hour",
            y="Trips",
            x_label="Pickup hour (0–23)",
            y_label="Number of trips"
        )

        busiest_hour = hourly_trips.loc[hourly_trips["Trips"].idxmax()]
        st.caption(
            f"Sabse busy hour {busiest_hour['Pickup hour']}:00 hai "
            f"({busiest_hour['Trips']} trips)."
        )

    with right:
        st.subheader("Cancellation rate by weekday")

        weekday_order = [
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        ]

        cancel_by_day = (
            filtered.assign(
                is_cancelled=filtered["status"] == "Cancelled"
            )
            .groupby("pickup_day")["is_cancelled"]
            .mean()
            .mul(100)
            .reindex(weekday_order)
            .dropna()
            .rename_axis("Weekday")
            .reset_index(name="Cancellation rate (%)")
        )

        st.bar_chart(
            cancel_by_day,
            x="Weekday",
            y="Cancellation rate (%)",
            x_label="Pickup weekday",
            y_label="Cancellation rate (%)"
        )

        if not cancel_by_day.empty:
            highest_day = cancel_by_day.loc[
                cancel_by_day["Cancellation rate (%)"].idxmax()
            ]
            st.caption(
                f"Sabse zyada cancellation {highest_day['Weekday']} ko: "
                f"{highest_day['Cancellation rate (%)']:.1f}%."
            )

    # Fare and distance chart
    st.divider()
    st.subheader("Fare vs distance — completed trips")

    if completed.empty:
        st.info("Fare-distance chart ke liye Completed status chunein.")
    else:
        sample = completed.sample(
            n=min(3000, len(completed)),
            random_state=42
        )

        st.scatter_chart(
            sample,
            x="distance_km",
            y="fare_amount",
            x_label="Trip distance (km)",
            y_label="Fare (source units)"
        )

        if completed["distance_km"].nunique() > 1:
            correlation = completed["distance_km"].corr(
                completed["fare_amount"]
            )
            st.caption(
                f"Selected completed trips mein fare-distance correlation "
                f"{correlation:.2f} hai. Ye association hai, cause-and-effect "
                "ka proof nahi."
            )


if 1:
    st.header("Key insights")
    st.caption("Ye findings selected filters ke hisaab se update hoti hain.")

    completed = filtered[filtered["status"] == "Completed"]
    completion_rate = (filtered["status"] == "Completed").mean() * 100
    cancellation_rate = (filtered["status"] == "Cancelled").mean() * 100
    no_show_rate = (filtered["status"] == "No-Show").mean() * 100

    st.markdown(
        f"1. **Trip outcomes:** {completion_rate:.1f}% completed, "
        f"{cancellation_rate:.1f}% cancelled, aur {no_show_rate:.1f}% no-show. "
        "Overview ke status chart mein comparison dekhein."
    )

    if not completed.empty:
        city_summary = (
            completed.groupby("city")
            .agg(
                trips=("trip_id", "size"),
                completed_fare=("fare_amount", "sum")
            )
        )
        top_trip_city = city_summary["trips"].idxmax()
        top_fare_city = city_summary["completed_fare"].idxmax()

        st.markdown(
            f"2. **City comparison:** sabse zyada completed trips "
            f"{top_trip_city} mein; completed fare total sabse zyada "
            f"{top_fare_city} mein."
        )

        if completed["distance_km"].nunique() > 1:
            correlation = completed["distance_km"].corr(
                completed["fare_amount"]
            )
            st.markdown(
                f"3. **Fare aur distance:** completed trips ka correlation "
                f"{correlation:.2f} hai. Overview ka scatter chart dekhein."
            )
        else:
            st.markdown(
                "3. **Fare aur distance:** is selection mein distance variation "
                "kam hai, isliye sambandh calculate nahi hua."
            )
    else:
        st.markdown(
            "2–3. Completed trips ka city aur fare comparison dekhne ke liye "
            "Completed status select karein."
        )

    hourly_counts = (
        filtered.groupby("pickup_hour").size().reindex(range(24), fill_value=0)
    )
    st.markdown(
        f"4. **Busy hours:** hourly trip counts {hourly_counts.min()} se "
        f"{hourly_counts.max()} tak hain. Overview ka hourly chart dekhein."
    )

    day_cancel = (
        filtered.assign(is_cancelled=filtered["status"] == "Cancelled")
        .groupby("pickup_day")["is_cancelled"]
        .mean()
        .mul(100)
        .dropna()
    )
    if not day_cancel.empty:
        st.markdown(
            f"5. **Weekday cancellations:** sabse zyada rate "
            f"{day_cancel.idxmax()} ko {day_cancel.max():.1f}% thi."
        )

    st.subheader("Recommendations")
    st.markdown(
        "1. Coordinates aur city labels ko verify karein; correction se pehle "
        "map-based decisions na lein."
    )
    st.markdown(
        "2. Cancellation patterns ko wait time aur driver availability ke saath "
        "investigate karein."
    )
    st.markdown(
        "3. Fare compare karte waqt similar distance wali completed trips dekhein "
        "aur source fare ki currency confirm karein."
    )

    st.info(
        "Ye observations sirf is supplied dataset ke baare mein hain. "
        "Coordinates aur bahut uniform patterns synthetic/example data ka "
        "sanket dete hain."
    )


    st.header("Key Insights")
    st.caption("Findings sidebar mein chune gaye filters ke hisaab se update hoti hain.")

    # 1. Trip outcomes
    completed = filtered[filtered["status"] == "Completed"]
    completed_rate = (filtered["status"] == "Completed").mean() * 100
    cancelled_rate = (filtered["status"] == "Cancelled").mean() * 100
    no_show_rate = (filtered["status"] == "No-Show").mean() * 100

    st.markdown(
        f"**1. Trip outcomes:** {completed_rate:.1f}% trips complete hui, "
        f"{cancelled_rate:.1f}% cancel hui, aur {no_show_rate:.1f}% no-show rahi. "
        "Overview ke *Trips by status* chart mein comparison dekhein."
    )

    # 2. City comparison
    if not completed.empty:
        city_summary = (
            completed.groupby("city")
            .agg(
                completed_trips=("trip_id", "size"),
                completed_fare=("fare_amount", "sum")
            )
        )

        top_trip_city = city_summary["completed_trips"].idxmax()
        top_fare_city = city_summary["completed_fare"].idxmax()

        st.markdown(
            f"**2. City comparison:** sabse zyada completed trips "
            f"**{top_trip_city}** mein ({city_summary.loc[top_trip_city, 'completed_trips']:,}); "
            f"completed fare total sabse zyada **{top_fare_city}** mein "
            f"({city_summary.loc[top_fare_city, 'completed_fare']:,.2f} source units)."
        )

        # City chart
        st.bar_chart(
            city_summary.reset_index(),
            x="city",
            y="completed_trips",
            x_label="City",
            y_label="Completed trips"
        )
    else:
        st.info("City aur fare findings ke liye Completed status select karein.")

    # 3. Fare aur distance
    if len(completed) > 1 and completed["distance_km"].nunique() > 1:
        correlation = completed["distance_km"].corr(completed["fare_amount"])
        slope = np.polyfit(
            completed["distance_km"],
            completed["fare_amount"],
            1
        )[0]

        st.markdown(
            f"**3. Fare aur distance:** completed trips ka correlation "
            f"**{correlation:.2f}** hai. Simple linear estimate mein har extra "
            f"km ke saath fare lagbhag **{slope:.2f} source units** badhta hai. "
            "Ye association hai, official pricing rule ya cause-and-effect ka proof nahi."
        )
    else:
        st.markdown(
            "**3. Fare aur distance:** is filter selection mein sambandh calculate "
            "karne ke liye paryapt distance variation nahi hai."
        )

    # 4. Busy hour
    hourly_counts = (
        filtered.groupby("pickup_hour")
        .size()
        .reindex(range(24), fill_value=0)
    )
    busiest_hour = hourly_counts.idxmax()

    st.markdown(
        f"**4. Busy hour:** sabse zyada trips **{busiest_hour}:00** baje hui "
        f"({hourly_counts.max():,} trips). Overview ke hourly chart mein baaki "
        "hours se compare karein."
    )

    # 5. Weekday cancellation aur payment method
    cancellation_by_day = (
        filtered.assign(
            is_cancelled=filtered["status"] == "Cancelled"
        )
        .groupby("pickup_day")["is_cancelled"]
        .mean()
        .mul(100)
        .dropna()
    )

    payment_counts = filtered["payment_method"].value_counts()

    if not cancellation_by_day.empty:
        highest_cancel_day = cancellation_by_day.idxmax()
        st.markdown(
            f"**5. Cancellations aur payments:** cancellation rate sabse zyada "
            f"**{highest_cancel_day}** ko thi "
            f"({cancellation_by_day.max():.1f}%). Sabse zyada use hua payment "
            f"method **{payment_counts.idxmax()}** tha "
            f"({payment_counts.max():,} trips). Overview mein weekday aur payment "
            "charts bhi dekhein."
        )

    st.subheader("Recommendations")
    st.markdown(
        "1. Coordinates aur city labels verify karein; tab tak map-based decisions na lein."
    )
    st.markdown(
        "2. Cancellation patterns ko wait time aur driver availability ke saath investigate karein."
    )
    st.markdown(
        "3. Fare compare karte waqt similar distance wali completed trips dekhein; "
        "source fare ki currency confirm karein."
    )

    st.caption(
        "Ye findings sirf selected dataset ke baare mein hain. Coordinates mein "
        "mismatch hai aur patterns synthetic/example data jaise lagte hain."
    )

    city_chart_data = city_summary.reset_index()

    left, right = st.columns(2)

    with left:
            st.markdown("#### Completed trips by city")
            st.bar_chart(
                city_chart_data,
                x="city",
                y="completed_trips",
                x_label="City",
                y_label="Completed trips"
            )
            st.caption(
                f"Top city: {top_trip_city} "
                f"({city_summary.loc[top_trip_city, 'completed_trips']:,} trips)"
            )

    with right:
            st.markdown("#### Completed fare total by city")
            st.bar_chart(
                city_chart_data,
                x="city",
                y="completed_fare",
                x_label="City",
                y_label="Fare (source units)"
            )
            st.caption(f"Top fare total: {top_fare_city}")