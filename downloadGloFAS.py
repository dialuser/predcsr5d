#author: Alex Sun
#date: 06262023
#purpose: download glofas reanalysis data
#====================================================================================
import os
import cdsapi

#fileroot = '/scratch/02248/alexsund/glofas'
fileroot = 'data/glofas'
def downloadGloFAS(startYear, endYear):
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
                'area': [
                    # north, west, south, east
                    #60, -130, 15,-40,   #north america
                    #10, -85, -55,  -40   #south america
                    #20, -15, -40,  60     #africa
                    70,  -10,   35, 60     #europe
                    #60, 30, 8, 145     #asia
                    #-10, 110, -40, 160     #australia

                ],                
            },
            os.path.join(fileroot, f'glofasv40_{iyear}_eu.grib'))

if __name__ == '__main__':    
    downloadGloFAS(2013, 2021)
