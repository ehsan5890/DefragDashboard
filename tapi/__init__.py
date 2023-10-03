# here goes all the code to work with TAPI
import json

import requests

from optical_rl_gym.utils import Service


class TAPIClient:
    url: str = "https://localhost:8082/restconf/data/tapi-common:context"
    authorization: str = "Basic YWRtaW46bm9tb3Jlc2VjcmV0"
    mock: bool = False

    def __init__(self, mock: bool = False) -> None:
        self.mock = mock
        self.__connect__()

    def __connect__(self) -> None:
        # test if a connection can be made
        if self.mock:
            return
        requests.get(
            self.url,
            headers={
                "Authorization": self.authorization,
            },
            verify=False,
            timeout=300,
        )
    
    def get_context(self) -> dict:
        if self.mock:
            with open("tests/context.json", "rt", encoding="UTF-8") as file:
                text = file.read()
            context = json.loads(text)
            return context
        
        context = requests.get(
            "https://localhost:8082/restconf/data/tapi-common:context",
            headers={
                "Authorization": "Basic YWRtaW46bm9tb3Jlc2VjcmV0"
            },
            verify=False,
            timeout=300,
        )
        return context.json()
    
    def get_services(self) -> list:
        # if self.mock:
        #     with open("tests/list_service.json", "rt", encoding="UTF-8") as file:
        #         text = file.read()
        #     context = json.loads(text)
        #     return context
        response = requests.get(
            "https://localhost:8082/restconf/data/tapi-common:context/tapi-connectivity:connectivity-context/connectivity-service",
            headers={
                "Authorization": "Basic YWRtaW46bm9tb3Jlc2VjcmV0",
            },
            verify=False,
            timeout=300,
        )
        services = response.json()
        service_mapping = {}
        for service in services['tapi-connectivity:connectivity-service']:
            found = False
            s_name = ""
            for name in service["name"]:
                # print(name)
                if "ODU" in name["value"] or "GBE100" in name["value"]:
                    continue
                found = True
                s_name = name["value"]
            if found:
                if s_name.endswith("_NFC"):
                    s_name = s_name.replace("_NFC", "")
                    if s_name not in service_mapping:
                        service_mapping[s_name] = {}
                    for endpoint in service["end-point"]:
                        service_mapping[s_name]["central-frequency"] = endpoint["tapi-adva:adva-connectivity-service-end-point-spec"]["adva-network-port-parameters"]["channel"]["central-frequency"]
                else:
                    # pprint(service)
                    # pprint(service["end-point"])
                    if s_name not in service_mapping:
                        service_mapping[s_name] = {"uuid": service["uuid"]}
                    else:
                        service_mapping[s_name]["uuid"] = service["uuid"]
                    for endpoint in service["end-point"]:
                        for key, value in endpoint["connection-end-point"][0].items():
                            if key not in service_mapping[s_name]:
                                service_mapping[s_name][key] = []
                            service_mapping[s_name][key].append(value)
        return service_mapping


    def establish_new_service(self, service: Service) -> bool:
        return True
    
    def delete_service(self, service: Service) -> bool:
        return True

    # ...
