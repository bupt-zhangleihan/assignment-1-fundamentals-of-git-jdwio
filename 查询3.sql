SELECT Cities.city,Airports.name,Cities.country
FROM Airports
    INNER JOIN Cities ON Cities.id= Airports.city_id