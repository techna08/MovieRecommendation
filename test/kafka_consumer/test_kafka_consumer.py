import sys
import os
import logging
from io import StringIO

sys.path.append(os.path.dirname(__file__) + "/../../")
from kafkaConsumer import getPath

def test_kafka_consumer():
    recom='2023-03-24T14:14:43.074025,773556,recommendation request 17645-team13.isri.cmu.edu:8082, status 200, result: the+shining+1980, interstellar+2014, harry+potter+and+the+goblet+of+fire+2005, pirates+of+the+caribbean+the+curse+of+the+black+pearl+2003, star+wars+1977, raiders+of+the+lost+ark+1981, monsters_+inc.+2001, forrest+gump+1994, the+green+mile+1999, harry+potter+and+the+deathly+hallows+part+2+2011, princess+mononoke+1997, 55 ms'
    mpg='2023-03-24T14:14:43,451866,GET /data/m/elysium+2013/90.mpg'
    ratings='2023-03-24T14:14:43,794772,GET /rate/separation+city+2009=4'
    path=getPath(recom)
    expected_path=['2023-03-24/1400-1500/ratings.txt', '2023-03-24/1400-1500/mpg.txt', '2023-03-24/1400-1500/recommendations.txt']
    assert path == expected_path, "Expected path: {}. Got path: {}.".format(expected_path, path)