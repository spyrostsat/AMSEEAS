import re
import win32com
import os
import numpy as np
from pydsstools.heclib.dss import HecDss


def hec_ras(new_solution):
    global total_overflows, total_costs, lateral_heights, counter_crown, solutions, not_keeping

    # new_solution is a 1D numpy array 1x27 with 27 different heights for the 27 different lateral structures
    # Throughout the whole length of a single lateral structure, we consider one mutual height for all cross sections

    lateral_width = 1  # meters
    cost = 1  # euros per m3 of additional lateral structure

    counter_crown = 0

    # Even counter_crown values represent the position of each cross-section across the total
    # length of each lateral structure, while odd values represent the height of the lateral
    # structure at the specific cross-section

    counter_laterals = -1
    # there are 27 (0 to 26) total lateral structures. At first, none of them is found, so the
    # counter_laterals variable is instantiated with -1

    def replace(match_object):
        global counter_crown
        if counter_crown % 2 == 0:
            counter_crown += 1
            return str(match_object.group(0))
        else:
            counter_crown += 1
            x = float(match_object.group(0)) + new_solution[counter_laterals]
            x = str(x)
            desired_length = len(str(match_object.group(0)))
            if len(x) == desired_length:
                return x
            else:
                while len(x) != desired_length:
                    if len(x) < desired_length:
                        x = x[:] + "0"
                    else:
                        x = x[:-1]
                return x

    with open("TRA_HEC_1D/TRA_L.g16true", "r") as f:
        contents = f.readlines()

    interested = False  # this variable determines whether a region of interest is found (i.e. lines of the file where
    # the lateral weirs are defined)

    counter = 0  # this counter counts which line of the geometry file I am currently reading
    for content in contents:
        if "Lateral Weir SE=" in content:
            counter_laterals += 1
            interested = True
            counter += 1
            continue

        if interested:
            if content[0].isalpha():
                interested = False
            else:
                new_string = re.sub(r"(\d+\.*\d*)", replace, content)
                contents[counter] = new_string
                counter += 1
                continue

        counter += 1

    contents = "".join(contents)

    with open("TRA_HEC_1D\TRA_L.g16", "w") as f:
        f.write(contents)

    # Initiate the RAS Controller class
    hec = win32com.client.Dispatch("RAS610.HECRASController")
    # hec.ShowRas()
    # full filename of the RAS project
    RASProject = os.path.join(os.getcwd(), "TRA_HEC_1D\TRA_L.prj")
    # opening HEC-RAS project
    hec.Project_Open(RASProject)
    calc_completed = False

    hec.Compute_CurrentPlan()

    while not calc_completed:
        if hec.Compute_Complete():
            calc_completed = True

    hec.QuitRas()  # this command closes the hec-ras window and also stops the simulation if it's not already completed

    with open("TRA_HEC_1D\TRA_L.p02.computeMsgs.txt", "r") as f:
        contents = f.readlines()
        for content in contents:
            if "The Model Has One Or More Error(s)" in content:
                print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                if not_keeping is False:
                    print("Keeping")
                    return np.random.uniform(low=0.65, high=0.80)
                else:
                    return 10 ** 10

    dss_file = "TRA_HEC_1D\TRA_L.dss"  # this is the path of the dss file being automatically

    pathnames = ['/TRA_EX TRA_EX/2705.61 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/2705.6 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/2305.6 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/1905.6 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/1714 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/1708.68 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/1690 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/1340.1 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/1308.68 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/908.68 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/890.1 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/508.68 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/440.1 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA_EX TRA_EX/108.68 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/1925.5 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/1565.1 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/1525.5 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/1125.5 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/1115.1 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/725.5 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/665.1 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/325.5 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_DO/221 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_UP/3078.41 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_UP/3078.4 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_UP/2678.4 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 '/TRA TRA_UP/2278.4 LAT STRUCT/FLOW-TOTAL/31Aug2000 - 01Sep2000/5MIN/PLAN 35/',
                 ]

    laterals_lengths = [106.2, 400, 400, 200, 5, 400, 90.05, 450, 400, 400, 450, 400, 397.8, 87.86, 400, 450, 400,
                        400, 450, 400, 409.59, 105, 6, 400, 255.57, 400, 335]

    fid = HecDss.Open(dss_file)

    total_overflow = 0  # We want to compute the total amount of overflow (in m3) across all lateral structures of the river
    total_cost = 0  # We also want to compute the total cost of the operation, i.e. the total cost of the additional
    # lateral structures we will build

    # The 2 parts of the objective function (i.e. total_overflow and total_cost) are conflicted, meaning that
    # the more height we give to the lateral structures, the less total_overflow will emerge, but the total_cost
    # will be higher

    for i in range(len(pathnames)):
        ts = fid.read_ts(pathnames[i], trim_missing=True)
        values = ts.values
        total_overflow += max(values.sum(), 0) * 60 * 5
        # the values are expressed in terms of m3/s and refer to a 5MIN plan, so we multiply those by first by
        # 60 (seconds) and then by 5 (min) to get the total overflow in m3.

        total_cost += laterals_lengths[i] * new_solution[i] * lateral_width * cost

    fid.close()

    max_overflow = 3272571.5  # the 'do nothing' solution (i.e. 0m elevation in all lateral structures) results in a
    # total overflow equal to 3272571.5 m3 and 0 cost
    max_elevation = 1  # we allow the elevation of all laterals to range from 0 to 1
    total_laterals_length = 8598.07  # the sum of the laterals_lengths list in meters
    max_cost = lateral_width * cost * max_elevation * total_laterals_length

    of = ((total_overflow / max_overflow) + (total_cost / max_cost)) / 2
    # of (i.e. objective function) values range from 0 to 1
    # we standardize both the overflow and the cost so that they both range from 0 to 1, and we consider as objective
    # function the average of these two standardized values

    total_overflows.append(total_overflow)
    total_costs.append(total_cost)
    lateral_heights.append(new_solution)
    solutions.append(of)

    return of
