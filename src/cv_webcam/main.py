def main():
    from cv_webcam.core.cv_proc import test_http

    url = "http://10.70.74.169:4747/video"
    test_http(url)


if __name__ == "__main__":
    main()
