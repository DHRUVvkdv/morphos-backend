import httpx
import asyncio


async def test_password_grant():
    domain = "dev-se4503ubrdrfcjal.us.auth0.com"
    client_id = "OoIKFCCAkLbB5b2mqDChtqAUnVq0cx6C"
    client_secret = "SmqZQ0KoIYtW8Vge4bi7EZXylfNoG4L1RvLyD-THeZ59QNDiUiUWLcawGjvWyf_D"

    # Test user credentials
    email = "dhruv@example.com"
    password = "Dhruv@12"

    print("Testing password grant...")
    async with httpx.AsyncClient() as client:
        # Method 1: Using form data (URL-encoded)
        print("\nTrying with form data:")
        response1 = await client.post(
            f"https://{domain}/oauth/token",
            data={
                "grant_type": "password",
                "username": email,
                "password": password,
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "openid profile email",
                "audience": f"https://{domain}/api/v2/",
                "realm": "Username-Password-Authentication",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        print(f"Status: {response1.status_code}")
        print(f"Response: {response1.text}")

        # Method 2: Using JSON
        print("\nTrying with JSON:")
        response2 = await client.post(
            f"https://{domain}/oauth/token",
            json={
                "grant_type": "password",
                "username": email,
                "password": password,
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "openid profile email",
                "audience": f"https://{domain}/api/v2/",
                "realm": "Username-Password-Authentication",
            },
        )
        print(f"Status: {response2.status_code}")
        print(f"Response: {response2.text}")

        # Method 3: Using /oauth/ro (legacy endpoint)
        print("\nTrying legacy /oauth/ro endpoint:")
        response3 = await client.post(
            f"https://{domain}/oauth/ro",
            json={
                "client_id": client_id,
                "username": email,
                "password": password,
                "connection": "Username-Password-Authentication",
                "grant_type": "password",
                "scope": "openid profile email",
            },
        )
        print(f"Status: {response3.status_code}")
        print(f"Response: {response3.text}")

        # Method 4: Using /oauth/token without audience
        print("\nTrying without audience:")
        response4 = await client.post(
            f"https://{domain}/oauth/token",
            data={
                "grant_type": "password",
                "username": email,
                "password": password,
                "client_id": client_id,
                "client_secret": client_secret,
                "scope": "openid profile email",
                "realm": "Username-Password-Authentication",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        print(f"Status: {response4.status_code}")
        print(f"Response: {response4.text}")


asyncio.run(test_password_grant())
