#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <fstream>

//int main() {
//    std::cout << "Hello, World!" << std::endl;
//    return 0;
//}

using namespace Eigen;

double kEps = 1e-10;

void
CalcF(double xi, double eta, double xk, double yk, double nkx, double nky, double L,
      double *F1, double *F2) {
    double A = pow(L, 2.0);
    double B = 2.0 * L * (-nky * (xk - xi) + nkx * (yk - eta));
    double E = pow(xk - xi, 2.0) + pow(yk - eta, 2.0);
    double M = sqrt(fabs(4.0 * A * E - pow(B, 2.0)));
    double BA = B / A;
    double EA = E / A;
    if (M < kEps) {
        *F1 = 0.5 / M_PI * L * (log(L) + (1 + 0.5 * BA) * log(fabs(1 + 0.5 * BA)) -
                                1 - 0.5 * BA * log(fabs(0.5 * BA)));
        *F2 = 0.0;
    } else {
        *F1 = 1.0 / M_PI * 0.25 * L * (2.0 * (log(L) - 1.0) - 0.5 * BA * log(fabs(EA)) +
                                       (1.0 + 0.5 * BA) * log(fabs(1.0 + BA + EA)) +
                                       (M / A) * (atan((2.0 * A + B) / M) - atan(B / M)));
        *F2 = 1.0 / M_PI * L * (nkx * (xk - xi) + nky * (yk - eta)) / M * (
                atan((2 * A + B) / M) - atan(B / M));
    }
}


void
CELAP1(int N,
       VectorXd *xm, VectorXd *ym,
       VectorXd *xb, VectorXd *yb,
       VectorXd *nx, VectorXd *ny,
       VectorXd *lg,
       VectorXd *BCT, VectorXd *BCV,
       VectorXd *phi, VectorXd *dphi) {

    VectorXd B = VectorXd::Zero(N);
    MatrixXd A = MatrixXd::Zero(N, N);
    *phi = VectorXd::Zero(N);
    *dphi = VectorXd::Zero(N);

    double F1, F2, delta;
    for (int m = 0; m < N; m++) {
        for (int k = 0; k < N; k++) {
            CalcF((*xm)[m], (*ym)[m], (*xb)[k], (*yb)[k], (*nx)[k], (*ny)[k], (*lg)[k], &F1, &F2);
            delta = (k == m) ? 1 : 0;
            if ((*BCT)[k] == 0) {
                A(m, k) = -F1;
                B[m] = B[m] + (*BCV)[k] * (-F2 + 0.5 * delta);
            } else {
                A(m, k) = F2 - 0.5 * delta;
                B[m] = B[m] + (*BCV)[k] * F1;
            }
        }
    }
    VectorXd Z = A.colPivHouseholderQr().solve(B);

    for (int m = 0; m < N; m++) {
        if ((*BCT)[m] == 0) {
            (*phi)[m] = (*BCV)[m];
            (*dphi)[m] = Z[m];
        } else {
            (*phi)[m] = Z[m];
            (*dphi)[m] = (*BCV)[m];
        }
    }

}

void CELAP2(int N, double xi, double eta, VectorXd *xb, VectorXd *yb, VectorXd *nx, VectorXd *ny,
            VectorXd *lg, VectorXd *phi, VectorXd *dphi, double *sum) {
    double F1, F2;
    for (int i = 0; i < N; i++) {
        CalcF(xi, eta, (*xb)[i], (*yb)[i], (*nx)[i], (*ny)[i], (*lg)[i], &F1, &F2);
        (*sum) += (*phi)[i] * F2 - (*dphi)[i] * F1;
    }
}

VectorXd linspace(double low, double high, int divs) {
    VectorXd v = VectorXd::Zero(divs);
    double delta = high - low;
    double step = delta / (divs - 1);
    for (int i = 0; i < divs; i++) {
        v[i] = low + step * i;
    }
    return v;
}

// [1 3 5] -> [1 3 5]
//            [1 3 5]
//            [1 3 5]
// [2 4 6] -> [2 2 2]
//            [4 4 4]
//            [6 6 6]
void meshgrid(VectorXd *x, VectorXd *y, MatrixXd *recvX, MatrixXd *recvY) {
    long xDim = x->size();
    long yDim = y->size();
    MatrixXd X = MatrixXd::Zero(yDim, xDim);
    MatrixXd Y = MatrixXd::Zero(yDim, xDim);
    for (long i = 0; i < yDim; i++) {
        X.row(i) = *x;
    }
    for (long i = 0; i < xDim; i++) {
        Y.col(i) = *y;
    }

    *recvX = X;
    *recvY = Y;
}

void mat_to_vec(MatrixXd *mat, std::vector<std::vector<double>> *vec) {
    for (int y = 0; y < mat->rows(); y++) {
        std::vector<double> tmp_vec;
        for (int x = 0; x < mat->cols(); x++) {
            tmp_vec.push_back((*mat)(y, x));
        }
        vec->push_back(tmp_vec);
    }
}

void vec_to_file(std::vector<std::vector<double>> *vec, std::string filename) {
    std::string csv = "";
    for (std::vector<double> i : *vec) {
        for (double j : i) {
            csv.append(std::to_string(j));
            csv.append(",");
        }
        csv.append("\n");
    }
    std::ofstream file(filename);
    std::cout << (file.is_open() ? "file open" : "reee") << std::endl;
    file << csv;
    file.close();
}

void mat_to_file(MatrixXd *mat, std::string filename) {
    std::vector<std::vector<double>> data;
    mat_to_vec(mat, &data);
    vec_to_file(&data, filename);
}


#define chksum

void print_vec(VectorXd *vec) {
    double sum = 0;
    for (int i = 0; i < vec->size(); i++) {
        std::cout << (*vec)(i) << ", ";
    }
    std::cout << std::endl;
}

void print_mat(MatrixXd *mat) {
    for (int i = 0; i < mat->rows(); i++) {
        for (int j = 0; j < mat->cols(); j++) {
            std::cout << round((*mat)(i, j) * 100) / 100 << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    std::cout << std::endl;
}


#ifndef fml

int main() {
    double u0 = 6.0;
    double Lx = 2.0;
    double Ly = 1.0;
    double lx = Lx;
    double ly = Ly / 2;
    double R = 0.3;
    double l_R = 1.0;
    int N0 = 100;
    int N = 5 * N0;
    double dlx = lx / N0;
    double dly = ly / N0;

    VectorXd xb = VectorXd::Zero(N + 1);
    VectorXd yb = VectorXd::Zero(N + 1);
    VectorXd xm = VectorXd::Zero(N);
    VectorXd ym = VectorXd::Zero(N);
    VectorXd lg = VectorXd::Zero(N);
    VectorXd nx = VectorXd::Zero(N);
    VectorXd ny = VectorXd::Zero(N);
    VectorXd BCT = VectorXd::Zero(N);
    VectorXd BCV = VectorXd::Zero(N);

    double N_R = 2 * N0 - (lx - 2 * R) / dlx;
    double i_b = (l_R - R) / dlx;
    double i_e = i_b + N_R;

    for (int i = 0; i < 2 * N0; i++) {
        if (i <= i_b) {
            xb[i] = i * dlx;
            yb[i] = 0;
        } else if ((i_b < i) && (i < i_e)) {
            xb[i] = l_R + R * cos(M_PI * (1 - (i - i_b) / N_R));
            yb[i] = sqrt(pow(R, 2) - pow(xb[i] - l_R, 2));
        } else {
            xb[i] = lx - (2 * N0 - i) * dlx;
            yb[i] = 0;
        }
    }

    for (int i = 0; i < N0; i++) {
        xb[2 * N0 + i] = lx;
        yb[2 * N0 + i] = i * dly;

        xb[3 * N0 + i] = N0 * dlx - i * dlx;
        yb[3 * N0 + i] = ly;

        xb[4 * N0 + i] = 0.0;
        yb[4 * N0 + i] = N0 * dly - i * dly;
    }

    xb[N] = xb[0];
    yb[N] = yb[0];

    //print_vec(&yb);
    //double sum_ = 0;
    //for (int i = 0; i < xb.size(); i++) {
    //    sum_ += yb(i);
    //}
    //std::cout << std::endl << sum_ << std::endl;
    //return 0;

    for (int i = 0; i < N; i++) {
        xm[i] = 0.5 * (xb[i] + xb[i + 1]);
        ym[i] = 0.5 * (yb[i] + yb[i + 1]);

        lg[i] = sqrt(pow(xb[i + 1] - xb[i], 2) + pow(yb[i + 1] - yb[i], 2));

        nx[i] = (yb[i + 1] - yb[i]) / lg[i];
        ny[i] = (-xb[i + 1] + xb[i]) / lg[i];
    }

    for (int i = 0; i < N; i++) {
        if (i < 2 * N0) {
            BCT[i] = 1;
            BCV[i] = 0;
        } else if (i < 3 * N0) {
            BCT[i] = 0;
            BCV[i] = 0;
        } else if (i < 4 * N0) {
            BCT[i] = 1;
            BCV[i] = 0;
        } else {
            BCT[i] = 1;
            BCV[i] = u0;
        }
    }

    VectorXd phi, dphi;
    CELAP1(N, &xm, &ym, &xb, &yb, &nx, &ny, &lg, &BCT, &BCV, &phi, &dphi);
    //print_vec(&xm);
    //print_vec(&ym);
    //print_vec(&xb);
    //print_vec(&yb);
    //print_vec(&nx);
    //print_vec(&ny);
    //print_vec(&lg);
    //print_vec(&BCT);
    //print_vec(&BCV);
    //print_vec(&phi);
    //print_vec(&dphi);
    //std::cout << xm.sum() << std::endl;
    //std::cout << ym.sum() << std::endl;
    //std::cout << xb.sum() << std::endl;
    //std::cout << yb.sum() << std::endl;
    //std::cout << nx.sum() << std::endl;
    //std::cout << ny.sum() << std::endl;
    //std::cout << lg.sum() << std::endl;
    //std::cout << BCT.sum() << std::endl;
    //std::cout << BCV.sum() << std::endl;
    //std::cout << phi.sum() << std::endl;
    //std::cout << dphi.sum() << std::endl;
    //return 0;

    int Nx = 4 * N0;
    int Ny = N0;
    double dx = lx / Nx;
    double dy = ly / Ny;
    VectorXd x = linspace(0, lx, Nx + 1);
    VectorXd y = linspace(0, ly, Ny + 1);
    MatrixXd X, Y;
    meshgrid(&x, &y, &X, &Y);
    MatrixXd phi_bem = MatrixXd::Zero(Ny + 1, Nx + 1);

    double sum;

    for (int i = 1; i < Ny; i++) {
        std::cout << i << " of " << Ny << std::endl;
        for (int j = 1; j < Nx; j++) {
            if (pow(X(i, j) - l_R, 2) + pow(Y(i, j), 2) > pow(R, 2)) {
                CELAP2(N, X(i, j), Y(i, j), &xb, &yb, &nx, &ny, &lg, &phi, &dphi, &phi_bem(i, j));
                //phi_bem(i, j) = sum;
                //std::cout << phi_bem(i, j) << std::endl;
            }
        }
    }
//print_mat(&phi_bem);

    for (int i = 1; i < Nx; i++) {
        phi_bem(0, i) = phi_bem(1, i);
        phi_bem(Ny, i) = phi_bem(Ny - 1, i);
    }

    for (int i = 1; i < Ny; i++) {
        phi_bem(i, 0) = phi_bem(i, 1) - u0 * dx;
        phi_bem(i, Nx) = 0;
    }


    MatrixXd u = MatrixXd::Zero(Ny + 1, Nx + 1);
    MatrixXd v = MatrixXd::Zero(Ny + 1, Nx + 1);

    int u_rows = (int) u.rows();
    int u_cols = (int) u.cols();
    int v_rows = (int) v.rows();
    int v_cols = (int) v.cols();
    //std::cout << phi_bem << std::endl;
    //std::cout << (phi_bem.rows() - 1) - 1 << " " << phi_bem.cols() - 2 << std::endl;

    // u[1:-1, 1:-1] = 0.5 / dx * (phi_bem[1:-1, 0:-2] - phi_bem[1:-1, 2:])
    // v[1:-1, 1:-1] = 0.5 / dy * (phi_bem[0:-2, 1:-1] - phi_bem[2:, 1:-1])

    auto *phi_chunk_u = new MatrixXd;
    auto *phi_chunk_v = new MatrixXd;
    auto *u_chunk = new MatrixXd;
    auto *v_chunk = new MatrixXd;

    *u_chunk = MatrixXd::Zero(u_rows - 1, u_cols - 1);
    *v_chunk = MatrixXd::Zero(u_rows - 1, u_cols - 1);
    *phi_chunk_u = MatrixXd::Zero(u_rows - 1, u_cols - 1);
    *phi_chunk_v = MatrixXd::Zero(v_rows - 1, v_rows - 1);

    long phi_rows = phi_bem.rows();
    long phi_cols = phi_bem.cols();

    *phi_chunk_u = phi_bem.block(1, 0, (phi_rows - 1) - 1, (phi_cols - 2) - 0) -
                   phi_bem.block(1, 2, (phi_rows - 1) - 1, (phi_cols - 0) - 2);
    *phi_chunk_v = phi_bem.block(0, 1, (phi_rows - 2) - 0, (phi_cols - 1) - 1) -
                   phi_bem.block(2, 1, (phi_rows - 0) - 2, (phi_cols - 1) - 1);


    *phi_chunk_u *= 0.5 / dx;
    *phi_chunk_v *= 0.5 / dy;
    *u_chunk = *phi_chunk_u;
    *v_chunk = *phi_chunk_v;

    u.block(1, 1, u_rows - 2, u_cols - 2) = *u_chunk;
    v.block(1, 1, v_rows - 2, v_cols - 2) = *v_chunk;

    delete (phi_chunk_u);
    delete (phi_chunk_v);
    delete (u_chunk);
    delete (v_chunk);



    //for (int col=0; col<u_cols; col++) {
    //    for (int row=0; col<u_rows; row++) {
    //        u(col, row) =
    //    }
    //}

    for (int i = 0; i < u.rows(); i++) {
        u(i, 0) = u0;
        v(i, 0) = 0;
    }
    for (int i = 0; i < u.cols(); i++) {
        u(0, i) = 0;
        v(0, i) = 0;
        v(Ny, i) = 0;
    }
    for (int i = 0; i < u.rows(); i++) {
        u(i, Nx) = u(i, Nx - 1);
        v(i, Nx) = v(i, Nx - 1);
    }

    //std::cout << u << std::endl;

    Matrix<double, 3, 3> A;
    A << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    MatrixXd e = A;
    MatrixXd f = A.array().pow(-1);
    //print_mat(&e);
    //print_mat(&f);

    print_mat(&u);

    mat_to_file(&u, "u.csv");
    mat_to_file(&v, "v.csv");
    mat_to_file(&X, "x.csv");
    mat_to_file(&Y, "y.csv");

}


#endif

#ifdef fml
int main() {
    VectorXd v = linspace(2, 3, 5);
    std::cout << v << std::endl << std::endl;

    VectorXd x = VectorXd::Zero(4);
    VectorXd y = VectorXd::Zero(3);
    x << 1, 3, 5, 7;
    y << 49, 3, 9;
    MatrixXd X;
    MatrixXd Y;
    meshgrid(&x, &y, &X, &Y);
    //for (int i=0; i<v.size(); i++) {
    //    std::cout << v[i] << std::endl;
    //}
    std::cout << x << std::endl << std::endl;
    std::cout << y << std::endl << std::endl;
    std::cout << X << std::endl << std::endl;
    std::cout << Y << std::endl << std::endl;

    return 0;
}
#endif
