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

    def establish_new_service(self, service: Service) -> bool:
        return True
    
    def delete_service(self, service: Service) -> bool:
        return True

    # ...
