# here goes all the code to work with TAPI
from optical_rl_gym.utils import Service


class TAPIClient:
    url: str

    def __init__(self) -> None:
        pass

    def __connect__(self) -> None:
        # test if a connection can be made
        pass

    def establish_new_service(self, service: Service) -> bool:
        return True
    
    def delete_service(self, service: Service) -> bool:
        return True

    # ...
