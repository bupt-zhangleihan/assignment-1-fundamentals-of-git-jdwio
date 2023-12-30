SELECT Cities.city,Airports.name
FROM Airports
    INNER JOIN Cities ON Cities.id= Airports.city_id
WHERE city = 'London'