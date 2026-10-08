import torch
from action_class import GetAction
import pandas as pd
from pathlib import Path

# mimic=Path(__file__).resolve().parent / "mimic" / "mimic-iii-clinical-database-demo-1.4"
doctor_values=[
    1.0,
    95.0,
    40.0,
    80.0,
    70.0,
    120.0,
    95.0,
    35.0,
    18.0,
    37.0,
    85.0,
    0.5,
    15.0,
    0.4,
    0.3,
    3.0,
    20.0,
    500.0,
    6.0,
    10.0,
    16.0,
    35.0,
    50.0,
    45.0,
    0.5,
    20.0
]

class Patient:
    pass
patient = Patient()
patient.age = 60
patient.sex = "Male"
patient.height = 175
patient.weight = 59

name_action = ["FiO2","Pinsp","Ti","RR","PEEP","Slope"]


def state_from_doctor(values):

    if len(values)!=26:
        raise ValueError(f"not 26 values,it's {len(values)}")

    state=torch.tensor(values,dtype=torch.float32)

    if not torch.isfinite(state).all():
        raise ValueError("inputs has NAN or missing ")

    return state


# def state_from_mimic(path,row_number=0):

#     df=pd.read_csv(path)
#     row=df.iloc[row_number]
#     values=row.to_numpy(dtype="float32")
#     if len(values)==27:
#         values=values[1:]

#     if len(values)!=26:
#         raise ValueError(f"Expected 27 state values, got {len(values)}")

#     state=torch.tensor(values,dtype=torch.float32)
#     if not torch.isfinite(state).all():
#         raise ValueError("mimics has NAN or missing ")

#     return state


agent=GetAction()

inputs=input("which way? doctor / mimic  : ").strip().lower()

if inputs=="doctor":

    state=state_from_doctor(doctor_values)

    print("State:")
    print(state)

    print("\nShape:")
    print(state.shape)

    print("GetAction loaded successfully.")
    print("\ndeciding which action is appropriate ...")
    action=agent.get_action(state, patient)
    print("\nraw action:")
    print(action)
    print("\nventilator settings:")
    for name, value in zip(name_action, action):
        print(f"{name}: {value.item():.3f}")


# elif inputs=="mimic":
#     state=state_from_mimic("patient.csv", row_number=0)
else:
    raise ValueError("Invalid INPUT_MODE. Enter 'doctor' or 'mimic'.")