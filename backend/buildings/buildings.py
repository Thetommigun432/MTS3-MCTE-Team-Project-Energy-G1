
class Building:
    def __init__(self, building_id, address, area, construction_year):
        self.building_id = building_id
        self.address = address
        self.area = area
        self.construction_year = construction_year
        self.appliances = []

    def get_building_info(self):
        return {
            "building_id": self.building_id,
            "address": self.address,
            "area": self.area,
            "construction_year": self.construction_year,
            "appliances": self.appliances
        }