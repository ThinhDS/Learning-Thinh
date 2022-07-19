Nhận diện biển số xe đi sử dụng thư viện OpenCV để tiền xử lý ảnh.
Grayscale chuyển đổi ảnh thành đa mức xám.
Dùng bộ lọc filter cho ảnh.
Thuật toán Canny hay Otsu để nhị phân hóa ảnh.
Dùng Contours để nhận diện vùng chứa biển số xe và thêm ROI để nhận diện từng ký tự và resize bằng với tập ký tự mẫu.
Cuối cùng, ta đi nhận diện ký tự bằng hệ số tương quan r-pearson với tập các ký tự mẫu cho trước.
