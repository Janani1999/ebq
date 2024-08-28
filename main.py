import streamlit as st
import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
from ortools.linear_solver import pywraplp
import datetime

# Streamlit UI
st.title("Yeos Monthly Production EBQ")

# Create a file uploader widget
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=['xlsx'])


if uploaded_file is not None:
    # Read the Excel file into different dataframes
    FC_ByMKT = pd.read_excel(uploaded_file, sheet_name='FC')  
    SS_YHSS = pd.read_excel(uploaded_file, sheet_name='SS_YHSS')
    SS_YHSM = pd.read_excel(uploaded_file, sheet_name='SS_YHSM&MY Co-Packer')
    SOH_MYWH = pd.read_excel(uploaded_file, sheet_name='MYWH_SOH')
    SOH_SGWH = pd.read_excel(uploaded_file, sheet_name='SGWH_SOH')
    cost = pd.read_excel(uploaded_file, sheet_name='Cost')
    logcost = pd.read_excel(uploaded_file, sheet_name='Transport Cost Mapped')
    production_param = pd.read_excel(uploaded_file, sheet_name='Production Parameter')
    
    
    # Replace NaN with 0s in all DataFrames
    FC_ByMKT.fillna(0, inplace=True)
    # YHSM_FC.fillna(0, inplace=True)
    # YHSM_SS.fillna(0, inplace=True)
    SS_YHSS.fillna(0, inplace=True)
    SS_YHSM.fillna(0, inplace=True)
    # YHSM_SOH.fillna(0, inplace=True)
    SOH_MYWH.fillna(0, inplace=True)
    SOH_SGWH.fillna(0, inplace=True)
    cost.fillna(0, inplace=True)
    logcost.fillna(0, inplace=True)
    # shelf_life.fillna(0, inplace=True)
    production_param.fillna(0, inplace=True)
    
    # Convert date columns to datetime objects and format to Year-Month
    FC_ByMKT.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in FC_ByMKT.columns]
    # YHSM_FC.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in YHSM_FC.columns]
    # YHSM_SS.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in YHSM_SS.columns]
    SS_YHSS.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in SS_YHSS.columns]
    SS_YHSM.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in SS_YHSM.columns]
    # SOH_MYWH.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in SOH_MYWH.columns]
    # SOH_SGWH.columns = [pd.to_datetime(col).strftime('%Y-%m') if isinstance(col, datetime.date) else col for col in SOH_SGWH.columns]
    
    
    ## User selects MFG, product type and formula sequentially
    # Get unique manufacturers
    manufacturers = FC_ByMKT['Manufacturer'].unique()
    
    # Sidebar selection boxes for manufacturer, product variant, and formula
    st.sidebar.subheader("Select Material")
    
    # Create columns for side-by-side selection boxes
    col1, col2, col3 = st.sidebar.columns(3)
    
    # Step 1: Select Manufacturer
    with col1:
        selected_manufacturer = st.selectbox('Select Manufacturer', manufacturers)
    
    # Filter DataFrame by selected manufacturer
    filtered_df_by_manufacturer = FC_ByMKT[FC_ByMKT['Manufacturer'] == selected_manufacturer]
    
    # Get unique product variants for the selected manufacturer
    product_variants = filtered_df_by_manufacturer['Product Variant'].unique()
    
    # Step 2: Select Product Variant
    with col2:
        selected_product_variant = st.selectbox('Select Product Variant', product_variants)
    
    # Filter DataFrame by selected product variant
    filtered_df_by_variant = filtered_df_by_manufacturer[filtered_df_by_manufacturer['Product Variant'] == selected_product_variant]
    
    # Get unique formulas for the selected product variant
    formulas = filtered_df_by_variant['Formula'].unique()
    
    # Step 3: Select Formula
    with col3:
        selected_formula = st.selectbox('Select Formula', formulas)

    
    # Year and Month range selectors
    st.sidebar.subheader("Select Date Range")
    years = [2024, 2025, 2026]
    months = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    
    start_year = st.sidebar.selectbox('Start Year', years, index=0)
    start_month = st.sidebar.selectbox('Start Month', months, index=6)
    end_year = st.sidebar.selectbox('End Year', years, index=0)
    end_month = st.sidebar.selectbox('End Month', months, index=10)
    
    start_date_str = f'{start_year}-{start_month}'
    end_date_str = f'{end_year}-{end_month}'
    # date_SOH_str = (pd.to_datetime(start_date_str) - pd.DateOffset(months=1)).strftime('%Y-%m')
    SS_start_date_str = (pd.to_datetime(start_date_str) + pd.DateOffset(months=1)).strftime('%Y-%m')
    SS_end_date_str = (pd.to_datetime(end_date_str) + pd.DateOffset(months=1)).strftime('%Y-%m')
    
        
    # Filter based on selected pdt var, mfg and formula
    filtered_df = FC_ByMKT[
        (FC_ByMKT["Product Variant"] == selected_product_variant ) &
        (FC_ByMKT["Manufacturer"] == selected_manufacturer) &
        (FC_ByMKT["Formula"] == selected_formula)
    ]
    
    #get the material ids
    material = filtered_df['Material']
    
    #get the countries
    country = filtered_df['Country'].unique()
    
        
    demand = filtered_df.loc[:,start_date_str:end_date_str].values
    demand = np.ceil(demand).astype(int)
    print('Demand forecast:', demand)
    M,N = demand.shape
    # # Round up the values to the nearest whole number
    # total_demand = quicksum(demand[i,t] for i in range(M) for t in range(N))
    total_demand = np.sum(demand)

    #concatenate SS_YHSS and SS_YHSM together
    SS = pd.concat([SS_YHSS, SS_YHSM], ignore_index=True)
    filtered_SS = SS[SS['Material'].isin(material)]
    SS_total = filtered_SS.loc[:, SS_start_date_str:SS_end_date_str].values.sum(axis=0)
    SS_total = SS_total.astype(int)
    
    # # Get Safety Stock data based on material id
    # filtered_SS_MYWH = SS_MYWH[SS_MYWH['Material'].isin(material)]
    # SS_MYWH_array = filtered_SS_MYWH.loc[:, SS_start_date_str:SS_end_date_str].values.sum(axis=0)
    # SS_MYWH_array = SS_MYWH_array.astype(int)
    
    # filtered_SS_SGWH = SS_SGWH[SS_SGWH['Material'].isin(material)]
    # SS_SGWH_array = filtered_SS_SGWH.loc[:, SS_start_date_str:SS_end_date_str].values.sum(axis=0)
    # SS_SGWH_array = SS_SGWH_array.astype(int)
    
    # SS_total = SS_MYWH_array + SS_SGWH_array
    
    #Get SOH based on the material id 
    SOH_MY = int(SOH_MYWH[SOH_MYWH['Material'].isin(material)].iloc[:, -1].sum())
    print('SOH in MY WH:', SOH_MY)
    SOH_SG = int(SOH_SGWH[SOH_SGWH['Material'].isin(material)].iloc[:, -1].sum())
    print('SOH in SG WH:', SOH_SG)
    SOH_Total = SOH_MY + SOH_SG
    
    # Dictionary to store averages and logistic cost for different countries
    avg_values = {}
    logcosts = {}
    
    #default average values (if neither SG or MY is present)
    avg_SG = 0
    avg_MY = 0
    inv_cost_SG = 0
    inv_cost_MY = 0

    filtered_cost = cost[
    (cost["Product Variant"] == selected_product_variant ) &
    (cost["Manufacturer"] == selected_manufacturer) &
    (cost["Formula"] == selected_formula)
    ]

    # Create columns for the inputs
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    for country in country:
        
        if country == 'Singapore':
            #Singapore demand
            demand_SG = filtered_df[(filtered_df['Country'] == country)].loc[:,start_date_str:end_date_str].values
            filtered_SS_SG = SS[SS['Country']==country]
            SS_SG_array = filtered_SS_SG.loc[:, SS_start_date_str:SS_end_date_str].values.sum(axis=0)
            SS_SG_array = SS_SG_array.astype(int)
            avg_SG = np.average(SS_SG_array) + np.average(demand_SG)
            avg_values[country] = avg_SG
            # Filter the logcost to Singapore
            logcost_SG = logcost[(logcost['Material'].isin(material)) & (logcost['Country'] == country)].iloc[:,-1].values[0]
            logcosts[country] = logcost_SG
            inv_cost_SG = filtered_cost.loc[:,'SGWH Inventory Holding Cost (SGD/ctn)'].values[0]
            with col1:
                inv_cost_SG = st.number_input('SG Warehouse Inventory Holding Cost (SGD/ctn)', value=inv_cost_SG, key='inv_cost_SG_key') #SGD
    
            
        elif country == 'Malaysia':
            #Demand in Malaysia
            demand_MY = filtered_df[(filtered_df['Country'] == country)].loc[:,start_date_str:end_date_str].values
            filtered_SS_MY = SS[SS['Country']==country]
            SS_MY_array = filtered_SS_MY.loc[:, SS_start_date_str:SS_end_date_str].values.sum(axis=0)
            SS_MY_array = SS_MY_array.astype(int)
            avg_MY = np.average(SS_MY_array) + np.average(demand_MY)
            avg_values[country] = avg_MY
            # Filter the logcost to Malaysia
            logcost_MY = logcost[(logcost['Material'].isin(material)) & (logcost['Country'] == country)].iloc[:,-1].values[0]
            logcosts[country] = logcost_MY
            # Obtain MY inventory cost
            inv_cost_MY = filtered_cost.loc[:,'MYWH Inventory Holding Cost (SGD/ctn)'].values[0]
            with col2:
                inv_cost_MY = st.number_input('MY Warehouse Inventory Holding Cost (SGD/ctn)', value= round(inv_cost_MY,2), key='inv_cost_MY_key') 
    
            
        else:
            demand_other = filtered_df[(filtered_df['Country'] == country)].loc[:,start_date_str:end_date_str].values
            print("Demand other:", demand_other)
            avg_other = np.average(demand_other)
            avg_values[country] = avg_other
            logcost_other = logcost[(logcost['Material'].isin(material)) & (logcost['Country'] == country)].iloc[:,-1].values[0]
            logcosts[country] = logcost_other
            

    frac_MY = round((avg_MY / (avg_MY + avg_SG)),1)
    print('frac_MY:', frac_MY)
    frac_SG = 1 - frac_MY
    weighted_avg_inv_cost = round((frac_MY * inv_cost_MY + frac_SG * inv_cost_SG),2)
    print('Weighted average inventory cost:', weighted_avg_inv_cost)
                
    
    # Calculate weighted average logistic transport cost
    # Calculate the total average value
    total_avg_value = sum(avg_values.values())
    # Initialize the weighted average logistic cost
    wavg_logcost = 0
    # Calculate the weighted average logistic cost
    for country, avg_value in avg_values.items():
        fraction = avg_value / total_avg_value
        logcost = logcosts.get(country, 0)
        wavg_logcost += fraction * logcost
        wavg_logcost = round(wavg_logcost,2)   

    #filter production param
    filtered_prod_param = production_param[
                (production_param["Product Variant"] == selected_product_variant) &
                (production_param["Manufacturer"] == selected_manufacturer) &
                (production_param["Formula"] == selected_formula)
            ]
    
    # Retrieve the Monthly Max Capacity, defaulting to None if not found
    monthly_max_capacity = int(filtered_prod_param.iloc[:, -1].values[0]) if not filtered_prod_param.iloc[:, -1].empty else None
    
    setup_cost = round((filtered_cost.loc[:, 'Fixed Setup Cost (SGD)'].values[0]),2) if not cost.loc[:, 'Fixed Setup Cost (SGD)'].empty else None
    
    variable_cost = round((filtered_cost.loc[:, 'Variable Cost (SGD/ctn)'].values[0]),2) if not cost.loc[:, 'Variable Cost (SGD/ctn)'].empty else None
    
    #batch_size
    # utilisation = 0.8  
    cooking_batch_size = int(filtered_prod_param.iloc[:, -2].values[0]) if not filtered_prod_param.iloc[:, -2].empty else None
    
    
    with col3:
        monthly_production_capacity = st.number_input('Monthly Production Capacity (ctn)', value=monthly_max_capacity) #allow user to key in
    with col4:
        fc = st.number_input('Fixed Cost (SGD)', value = setup_cost)
    with col5:
        vc = st.number_input('Variable Cost (SGD/ctn)', value=variable_cost)
    with col6:
        batch_size = st.number_input('Batch Size (ctn)', value=cooking_batch_size)
    
    shelf_life = filtered_df["Shelf Life (month)"].unique()
    shelf_life = shelf_life[0]
    
    with col7:
        shelf_life = st.number_input('Shelf Life (month)', value = shelf_life)
    
    prod_shelf_life = int( shelf_life - 6)
    
    
    ##### Optimise #####
    if st.button('Optimize'):
    
        # Initialize the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("Solver not created.")
        
        # Define decision variables
        z = [solver.BoolVar(f'z[{t}]') for t in range(N)]                       # To open production or not
        y = [solver.IntVar(0, solver.infinity(), f'y[{t}]') for t in range(N)]  # Production quantity
        w = [solver.IntVar(0, solver.infinity(), f'w[{t}]') for t in range(N)]  # Number of batches
        x = [solver.IntVar(0, solver.infinity(), f'x[{t}]') for t in range(N)]  # Unsold inventory quantity
        non_prod = [solver.IntVar(0, solver.infinity(), f'non_prod[{t}]') for t in range(N)]  #counter for number of non-production months
        
        # Objective: minimize production cost and inventory holding cost
        objective = solver.Objective()
        for t in range(N):
            objective.SetCoefficient(y[t], float(vc))                       
            objective.SetCoefficient(x[t], float(weighted_avg_inv_cost))     
            objective.SetCoefficient(z[t], float(fc))                      
        objective.SetMinimization()
        
        # Constraints
        # Demand constraint
        for t in range(1, N):
            solver.Add(x[t] == x[t - 1] + y[t] - sum(demand[j, t] for j in range(M)))
        solver.Add(x[0] == SOH_Total + y[0] - sum(demand[j, 0] for j in range(M)))
        
        # Safety stock constraint
        for t in range(N):
            solver.Add(x[t] >= SS_total[t])       
        
        # Production constraint
        for t in range(N):
            solver.Add(y[t] <= total_demand * z[t])
            solver.Add(y[t] <= monthly_production_capacity)
            solver.Add(y[t] == batch_size * w[t])
        
        # Consecutive non-production constraint
        for t in range(N):
            if t == 0:
                solver.Add(non_prod[t] == 1 - z[t])
            else:
                solver.Add(non_prod[t] == non_prod[t-1] + 1 - z[t])
            solver.Add(non_prod[t] <= prod_shelf_life)
        
        # Positivity constraint
        for t in range(N):
            solver.Add(y[t] >= 0)
            solver.Add(x[t] >= 0)
        
        # Solve the model
        status = solver.Solve()
        
        # Prepare the data for the table
        variables = ['setup', 'production_qty', 'number_of_cooking_batch', 'month_end_SOH']
        dates = pd.date_range(start=start_date_str, periods=N, freq='MS').strftime('%Y-%m')
        data = {
            'setup': [z[t].solution_value() for t in range(N)],
            'production_qty': [y[t].solution_value() for t in range(N)],
            'number_of_batch': [w[t].solution_value() for t in range(N)],
            'month_end_SOH': [x[t].solution_value() for t in range(N)],
            'date': dates
        }
        
        # Convert to DataFrame for display
        optimal_df = pd.DataFrame(data).set_index('date').T
    
        # Store results in session state
        st.session_state['obj_val'] = solver.Objective().Value()
        st.session_state['optimal_df'] = optimal_df
        
    # Display optimal solutions
    if 'optimal_df' in st.session_state:
        st.subheader("Optimal Solution")
        st.table(st.session_state['optimal_df'])
    
    # Display minimal total cost
    if 'obj_val' in st.session_state:
        st.write(f"Minimal Total Cost (SGD): {st.session_state['obj_val']: .2f}")
        # st.write(f"Objective Value: {st.session_state['obj_val']: .2f}")
    
    
    ##### Allow user to input the number of batches for each month #####
    st.subheader("Self Input Number of Production Batches for Each Month")
    
    #show the initial inventory and batch_size
    st.write(f"Current SOH (ctn): {SOH_Total}")
    st.write(f"Batch Size (ctn): {batch_size}")
    
    
    # Show the forecast and SS data in a table
    demand_dates = pd.date_range(start=start_date_str, periods=N, freq='MS').strftime('%Y-%m')
    agg_demand = [sum(demand[j, t] for j in range(M)) for t in range(N)]
    # agg_demand = demand.sum(axis=1)
    demand_data = {
        'Demand Forecast': agg_demand,
        'date': demand_dates
    }
    
    SS_dates = pd.date_range(start=SS_start_date_str, periods=N, freq='MS').strftime('%Y-%m')
    SS_data = {
        'Safety Stock': SS_total.astype(int).tolist(),
        'date': SS_dates
    }
    
    # Convert to DataFrame for display
    demand_df = pd.DataFrame(demand_data).set_index('date').T
    SS_df = pd.DataFrame(SS_data).set_index('date').T
    
    # Display the parameter table
    st.table(demand_df)
    st.table(SS_df)
    
    user_batches = []
    cols = st.columns(N)
    for i in range(N):
        with cols[i]:
            user_batches.append(st.number_input(f'{pd.date_range(start=start_date_str, periods=N, freq="MS")[i].strftime("%Y-%m")}', min_value=0, value=0))
    
    # agg_demand = [sum(demand[j, t] for j in range(M)) for t in range(N)]
    
    # Calculate and display results based on user inputs
    if st.button('Calculate Based on User Input'):
        user_production_qty = [batch_size * user_batches[i] for i in range(N)]
        inventory_balance = [0] * N
        inventory_balance[0] = SOH_Total + user_production_qty[0] - agg_demand[0]
        
        for t in range(1, N):
            inventory_balance[t] = inventory_balance[t - 1] + user_production_qty[t] - agg_demand[t]
    
        user_total_cost = sum((vc+wavg_logcost) * user_production_qty[t] + weighted_avg_inv_cost * inventory_balance[t] + (fc if user_batches[t] > 0 else 0) for t in range(N))
    
        #convert result values to integers
        user_production_qty = [int(val) for val in user_production_qty]
        user_batches = [int(val) for val in user_batches]  
        inventory_balance = [int(val) for val in inventory_balance]  
        
        # Prepare the data for the table
        user_data = {
            'setup': [1 if user_batches[t] > 0 else 0 for t in range(N)],
            'production_qty': user_production_qty,
            'number_of_batch': user_batches,
            'month_end_SOH': inventory_balance,
            'safety stock': SS_total.astype(int).tolist(),
            'Date': demand_dates
        }
    
        # Convert to DataFrame for display
        user_df = pd.DataFrame(user_data).set_index('Date').T
    
        # Highlight function
        def highlight_inventory(val, threshold):
            color = 'red' if val < threshold else ''
            return f'background-color: {color}'
        
         # Apply highlighting to the 'month_end_SOH' row
        def apply_highlighting(df, row_name, threshold_row_name):
            styles = pd.DataFrame(index=df.index, columns=df.columns)
            for col in df.columns:
                for row in df.index:
                    if row == row_name:
                        val = df.loc[row, col]
                        threshold = df.loc[threshold_row_name, col]
                        styles.loc[row, col] = highlight_inventory(val, threshold)
                    else:
                        styles.loc[row, col] = ''
                        
            return df.style.apply(lambda _: styles, axis=None)
    
        styled_user_df = apply_highlighting(user_df, 'month_end_SOH', 'safety stock')
    
        # Store results in session state
        st.session_state['user_total_cost'] = user_total_cost
        st.session_state['styled_user_df'] = styled_user_df
    
    # Display user inputs in a table
    if 'styled_user_df' in st.session_state:
        st.write("Results Based on User Input")
        st.dataframe(st.session_state['styled_user_df'])
    
    # Display total cost based on user inputs
    if 'user_total_cost' in st.session_state:
        st.write(f"Total Cost Based on User Input (SGD): {st.session_state['user_total_cost']:.2f}")
    
    #################################
    st.subheader("Summary")
    
    # Check if the necessary data is in the session state
    if 'optimal_df' in st.session_state and 'styled_user_df' in st.session_state:
        # Create two columns
        col1, col2 = st.columns([1,1])
    
        ### Append safety stock as new row to optimal_df ###
        new_row = SS_total.astype(int).tolist()
    
         # Create a DataFrame from the new row with the same columns as optimal_df
        new_row_df = pd.DataFrame([new_row], columns=st.session_state['optimal_df'].columns, index = ['safety stock'])
        
        # Append the new row to optimal_df
        st.session_state['optimal_df'] = pd.concat([st.session_state['optimal_df'], new_row_df], ignore_index=False)
    
        # Display optimal solution in the first column
        with col1:
            st.write("Optimal Solution")
            st.table(st.session_state['optimal_df'])
    
            if 'obj_val' in st.session_state:
                st.write(f"Minimal Total Cost (SGD): {st.session_state['obj_val']: .2f}")
    
        # Display user input result in the second column
        with col2:
            st.write("User Input Result")
            st.dataframe(st.session_state['styled_user_df'])
    
            if 'user_total_cost' in st.session_state:
                st.write(f"Total Cost (SGD): {st.session_state['user_total_cost']:.2f}")
else:
    st.write("No file uploaded yet.")
                    
