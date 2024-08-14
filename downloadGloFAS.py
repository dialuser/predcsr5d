#Author: Alex Sun
#Date: 06262023
#Purpose: download glofas reanalysis data
#====================================================================================
import os
import cdsapi

#set folder for data
fileroot = 'data/glofas'
def downloadGloFAS(startYear, endYear, region_name, region_box):
    c = cdsapi.Client()

    for iyear in range(startYear, endYear+1):
        c.retrieve(
            'cems-glofas-historical',
            {
                'system_version': 'version_4_0',
                'variable': 'river_discharge_in_the_last_24_hours',
                'format': 'grib',
                'hydrological_model': 'lisflood',
                'product_type': 'consolidated',
                'hyear': f'{iyear}',
                'hmonth': [
                    'april', 'august', 'december',
                    'february', 'january', 'july',
                    'june', 'march', 'may',
                    'november', 'october', 'september',
                ],
                'hday': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                    '13', '14', '15',
                    '16', '17', '18',
                    '19', '20', '21',
                    '22', '23', '24',
                    '25', '26', '27',
                    '28', '29', '30',
                    '31',
                ],
                'area': region_box,                
            },
            os.path.join(fileroot, f'glofasv40_{iyear}_{region_name}.grib'))

if __name__ == '__main__':    
    # north, west, south, east
    regions={
        'na': [60, -130, 15,-40],
        'sa': [10, -85, -55,  -40],
        'af': [20, -15, -40,  60],
        'eu': [70,  -10,   35, 60],
        'as': [60, 30, 8, 145],
        'au': [-10, 110, -40, 160 ]
    }
    for key in regions.keys():
        downloadGloFAS(2002, 2021, region_name=key, region_box=regions[key])
