import numpy as np
from scipy.special import gamma, kv



class AIEM:
    ERROR = 1e-10


    def __init__(self, frq_ghz, theta_i, theta_s, phi_i, phi_s, sigma, cl, eps, itype) -> None:

        self.er = eps
        self.ur = 1
        theta_i = np.deg2rad(theta_i)
        theta_s = np.deg2rad(theta_s)
        phi_i = np.deg2rad(phi_i)
        phi_s = np.deg2rad(phi_s)
        self.itype = itype

        frq_hz = frq_ghz * 1.0e9                                # Transform from GHz to Hz.
        sigma = sigma * 1.0e2                                   # Convert from m to cm.
        cl = cl * 1.0e2                                         # Convert from m to cm.
        k = 2 * np.pi * frq_hz / (299792458 * 1e2)                         # Wavenumer rad/cm
        self.ks = k * sigma                                     # roughness parameter
        self.kl = k * cl

        self.si = np.sin(theta_i)
        self.sis = np.sin(theta_s)
        self.sfi = np.sin(phi_i)
        self.sfs = np.sin(phi_s)
        self.cs = np.cos(theta_i)
        self.css = np.cos(theta_s)
        self.csfi = np.cos(phi_i)
        self.csfs = np.cos(phi_s)

        self.cs2 = self.cs**2
        self.css2 = self.css**2
        self.si2 = self.si**2
        self.sis2 = self.sis**2
        self.ks2 = self.ks**2
        self.kl2 = self.kl**2

        # Compute roughness spectrum spectra_1(n)
        self.iterm, self.spectra_1 = self.compute_roughness_spectrum()

        # Reflection coefficients based on the incident angle and the specular angle
        rvi, rhi, rvhi, rvl, rhl, rvhl = self.compute_reflection_coefficients()

        # Reflection coefficients based on the transition function R_transition (T.D.Wu&A.K.fung) for kirchhoff field coefficients         
        rv, rh, rvh, rhv = self.compute_transition_coefficients(rvi, rhi, rvl, rhl)

        # Kirchhoff field coefficients fvv, fhh, fhv, fvh
        fhh, fvv, fhv, fvh = self.kirchhoff_field_coefficients(rv, rh, rvh)

        # Compute scattering coefficients!
        Ivv, Ihh, Ihv, Ivh, CIvv, CIhh, CIhv, CIvh = self.compute_scattering_coefficients(rv, rh, rvh, rhv, fvv, fhh, fhv, fvh)

        # Compute backscattering from the surface
        self.VV, self.HH, self.HV, self.VH = self.compute_sigma0(Ivv, CIvv, Ihh, CIhh, Ihv, CIhv, Ivh, CIvh)


    def run(self):
        return self.HH, self.VH, self.HV, self.VV


    def compute_roughness_spectrum(self):
        
        torlant = 1.0e-16
        iterm = 1
        tempold = 0
        temp = self.ks2 * (self.cs + self.css) ** 2

        while (np.abs(temp - tempold) > torlant):
            tempold = temp
            iterm = iterm + 1
            fiterm = iterm
            temp = tempold * (self.ks2 * (self.cs + self.css) ** 2) / fiterm

        iterm += 1
        spectra_1 = np.zeros(iterm)

        for n in range(1, iterm):
            fn = n
            K = self.kl * np.sqrt((self.sis * self.csfs - self.si * self.csfi) ** 2 + (self.sis * self.sfs - self.si * self.sfi) ** 2)
    
            if self.itype == '1':
                # Gaussian correlated surface
                spectra_1[n-1] = self.kl2 * np.exp(-K * K / (4 * fn)) / (2 * fn)
            elif self.itype == '2':
                # Exponential correlated surface
                spectra_1[n-1] = (self.kl / fn) ** 2 * (1 + (K / fn) ** 2) ** (-1.5)
            elif self.itype == '3':
                # 1.5 power surface
                e = 1.5 * fn - 1
                y = 1.5 * fn
                gam = gamma(y)  # Logarithm of the gamma function (1.5n)
                if K == 0:
                    spectra_1[n-1] = self.kl * self.kl / (3 * fn - 2)
                else:
                    m = 1.5 * fn - 1
                    bk = np.log(kv(-m, K))
                    out = self.kl * self.kl * (K / 2) ** e
                    spectra_1[n-1] = out * np.exp(bk - gam)

        return iterm, spectra_1


    def compute_reflection_coefficients(self):
        
        # Reflection coefficients based on the incident angle
        stem = np.sqrt(self.er * self.ur - self.si ** 2)
        rvi = (self.er * self.cs - stem) / (self.er * self.cs + stem)
        rhi = (self.ur * self.cs - stem) / (self.ur * self.cs + stem)
        rvhi = (rvi - rhi) / 2.0

        # Reflection coefficients based on the specular angle
        csl = np.sqrt(1.0 + self.cs * self.css - self.si * self.sis * self.csfs) / np.sqrt(2.0)
        sil = np.sqrt(1.0 - csl ** 2)
        steml = np.sqrt(self.er * self.ur - sil ** 2)
        rvl = (self.er * csl - steml) / (self.er * csl + steml)
        rhl = (self.ur * csl - steml) / (self.ur * csl + steml)
        rvhl = (rvl - rhl) / 2.0

        return rvi, rhi, rvhi, rvl, rhl, rvhl


    def compute_transition_coefficients(self, rvi, rhi, rvl, rhl):

        # Reflection coefficients based on the transition function R_transition (T.D.Wu&A.K.fung)
        # Wu, T. D., Chen, K. S., Shi, J., & Fung, A. K. (2001). A transition model for the reflection coefficient in surface scattering. 
        # IEEE Transactions on Geoscience and Remote Sensing, 39(9), 2040-2050. https://doi.org/10.1109/36.951094
        # Tzong-Dar Wu and Kun-Shan Chen, "A reappraisal of the validity of the IEM model for backscattering from rough surfaces," 
        # in IEEE Transactions on Geoscience and Remote Sensing, vol. 42, no. 4, pp. 743-753, April 2004, doi: 10.1109/TGRS.2003.815405.
        
        rv0 = (np.sqrt(self.er) - 1.0) / (np.sqrt(self.er) + 1.0)
        rh0 = -(np.sqrt(self.er) - 1.0) / (np.sqrt(self.er) + 1.0)

        Ftv = 8.0 * (rv0 ** 2) * self.si2 * (self.cs + np.sqrt(self.er - self.si2)) / (self.cs * (np.sqrt(self.er - self.si2)))
        Fth = -8.0 * (rh0 ** 2) * self.si2 * (self.cs + np.sqrt(self.er - self.si2)) / (self.cs * (np.sqrt(self.er - self.si2)))

        st0v = 1.0 / (np.abs(1.0 + 8.0 * rv0 / (self.cs * Ftv)) ** 2.0)
        st0h = 1.0 / (np.abs(1.0 + 8.0 * rh0 / (self.cs * Fth)) ** 2.0)

        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        temp1 = 1.0

        for n in range(1, self.iterm):
            fn = n
            temp1 *= 1.0 / fn
            sum1 += temp1 * ((self.ks * self.cs) ** (2.0 * fn)) * self.spectra_1[n - 1]
            sum2 += temp1 * ((self.ks * self.cs) ** (2.0 * fn)) * (
                        np.abs(Ftv + 2.0 ** (fn + 2.0) * rv0 / self.cs / (np.exp((self.ks * self.cs) ** 2.0)))) ** 2.0 * self.spectra_1[n - 1]
            sum3 += temp1 * ((self.ks * self.cs) ** (2.0 * fn)) * (
                        np.abs(Fth + 2.0 ** (fn + 2.0) * rh0 / self.cs * (np.exp(-(self.ks * self.cs) ** 2.0)))) ** 2.0 * self.spectra_1[n - 1]

        stv = (np.abs(Ftv) ** 2.0) * sum1 / sum2
        sth = (np.abs(Fth) ** 2.0) * sum1 / sum3

        tfv = 1.0 - stv / st0v
        tfh = 1.0 - sth / st0h

        tfv = np.where(tfv < 0, 0, tfv)
        tfh = np.where(tfh < 0, 0, tfh)

        rvtran = rvi + (rvl - rvi) * tfv
        rhtran = rhi + (rhl - rhi) * tfh
        rvhtran = (rvtran - rhtran) / 2.0
        rhvtran = rvhtran

        return rvtran, rhtran, rvhtran, rhvtran


    def kirchhoff_field_coefficients(self, rv, rh, rvh):
        
        zxx = -(self.sis * self.csfs - self.si) / (self.css + self.cs)
        zyy = -(self.sis * self.sfs) / (self.css + self.cs)
        d2 = np.sqrt((zxx * self.cs - self.si) ** 2 + zyy ** 2)
        hsnv = -(self.cs * self.csfs + self.si * (zxx * self.csfs + zyy * self.sfs))
        vsnh = self.css * self.csfs - zxx * self.sis
        hsnh = -self.sfs
        vsnv = zyy * self.cs * self.sis + self.css * (zyy * self.csfs * self.si - (self.cs + zxx * self.si) * self.sfs)
        hsnt = (-(self.cs ** 2 + self.si ** 2) * self.sfs * (-self.si + self.cs * zxx) + self.csfs * (self.cs + self.si * zxx) * zyy + self.si * self.sfs * (zyy ** 2)) / d2
        hsnd = (-(self.cs + self.si * zxx) * (-self.csfs * self.si + self.cs * self.csfs * zxx + self.cs * self.sfs * zyy)) / d2
        vsnt = ((self.cs ** 2 + self.si ** 2) * (-self.si + self.cs * zxx) * (self.csfs * self.css - self.sis * zxx) + self.css * self.sfs * (self.cs + self.si * zxx) * zyy - (self.csfs * self.css * self.si + self.cs * self.sis) * (zyy ** 2)) / d2
        vsnd = -(self.cs + self.si * zxx) * (self.si * self.sis * zyy - self.css * (self.si * self.sfs - self.cs * self.sfs * zxx + self.cs * self.csfs * zyy)) / d2

        fhh = (1 - rh) * hsnv + (1 + rh) * vsnh - (hsnt + vsnd) * (rh + rv) * (zyy / d2)
        fvv = -((1 - rv) * hsnv + (1 + rv) * vsnh) + (hsnt + vsnd) * (rh + rv) * (zyy / d2)
        fhv = -(1 + rv) * hsnh + (1 - rv) * vsnv + (hsnd - vsnt) * (rh + rv) * (zyy / d2)
        fvh = -(1 + rh) * hsnh + (1 - rh) * vsnv + (hsnd - vsnt) * (rh + rv) * (zyy / d2)

        return fhh, fvv, fhv, fvh


    def compute_scattering_coefficients(self, rv, rh, rvh, rhv, fvv, fhh, fhv, fvh):
        
        # Reflection coefficients rv, rh, rhv for complementary field coefficients 
        qq = self.cs
        qqt = np.sqrt(self.er - self.si2)
        qqs = self.css
        qqts = np.sqrt(self.er - self.sis2)

        qq1 = qq
        qq2 = qqs
        qq3 = qqt
        qq4 = qqts
        qq5 = qqt
        qq6 = qqts

        # Fvv
        Fvaupi = self.favv(-self.si, 0.0, qq1, qq1, qq, rv) * self.expal(qq1)
        Fvadni = self.favv(-self.si, 0.0, -qq1, -qq1, qq, rv) * self.expal(-qq1)
        Fvaups = self.favv(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rv) * self.expal(qq2)
        Fvadns = self.favv(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rv) * self.expal(-qq2)
        Fvbupi = self.fbvv(-self.si, 0.0, qq3, qq5, qqt, rv) * self.expal(qq5)
        Fvbdni = self.fbvv(-self.si, 0.0, -qq3, -qq5, qqt, rv) * self.expal(-qq5)
        Fvbups = self.fbvv(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rv) * self.expal(qq6)
        Fvbdns = self.fbvv(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rv) * self.expal(-qq6)

        # Fhh
        Fhaupi = self.fahh(-self.si, 0.0, qq1, qq1, qq, rh) * self.expal(qq1)
        Fhadni = self.fahh(-self.si, 0.0, -qq1, -qq1, qq, rh) * self.expal(-qq1)
        Fhaups = self.fahh(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rh) * self.expal(qq2)
        Fhadns = self.fahh(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rh) * self.expal(-qq2)
        Fhbupi = self.fbhh(-self.si, 0.0, qq3, qq5, qqt, rh) * self.expal(qq5)
        Fhbdni = self.fbhh(-self.si, 0.0, -qq3, -qq5, qqt, rh) * self.expal(-qq5)
        Fhbups = self.fbhh(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rh) * self.expal(qq6)
        Fhbdns = self.fbhh(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rh) * self.expal(-qq6)

        # Fhv
        Fhvaupi = self.fahv(-self.si, 0.0, qq1, qq1, qq, rhv) * self.expal(qq1)
        Fhvadni = self.fahv(-self.si, 0.0, -qq1, -qq1, qq, rhv) * self.expal(-qq1)
        Fhvaups = self.fahv(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rhv) * self.expal(qq2)
        Fhvadns = self.fahv(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rhv) * self.expal(-qq2)
        Fhvbupi = self.fbhv(-self.si, 0.0, qq3, qq5, qqt, rhv) * self.expal(qq5)
        Fhvbdni = self.fbhv(-self.si, 0.0, -qq3, -qq5, qqt, rhv) * self.expal(-qq5)
        Fhvbups = self.fbhv(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rhv) * self.expal(qq6)
        Fhvbdns = self.fbhv(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rhv) * self.expal(-qq6)

        # Fvh
        Fvhaupi = self.favh(-self.si, 0.0, qq1, qq1, qq, rvh) * self.expal(qq1)
        Fvhadni = self.favh(-self.si, 0.0, -qq1, -qq1, qq, rvh) * self.expal(-qq1)
        Fvhaups = self.favh(-self.sis * self.csfs, -self.sis * self.sfs, qq2, qq2, qqs, rvh) * self.expal(qq2)
        Fvhadns = self.favh(-self.sis * self.csfs, -self.sis * self.sfs, -qq2, -qq2, qqs, rvh) * self.expal(-qq2)
        Fvhbupi = self.fbvh(-self.si, 0.0, qq3, qq5, qqt, rvh) * self.expal(qq5)
        Fvhbdni = self.fbvh(-self.si, 0.0, -qq3, -qq5, qqt, rvh) * self.expal(-qq5)
        Fvhbups = self.fbvh(-self.sis * self.csfs, -self.sis * self.sfs, qq4, qq6, qqts, rvh) * self.expal(qq6)
        Fvhbdns = self.fbvh(-self.sis * self.csfs, -self.sis * self.sfs, -qq4, -qq6, qqts, rvh) * self.expal(-qq6)

        # Compute scattering coefficients!
        Ivv, Ihh, Ihv, Ivh = (np.empty((self.iterm,) + self.er.shape, dtype=np.complex_) for _ in range(4))
        CIvv, CIhh, CIhv, CIvh = (np.empty((self.iterm,) + self.er.shape, dtype=np.complex_) for _ in range(4))
        
        for n in range(1, self.iterm):
            fn = n
            Ivv[n-1] = ((self.cs+self.css)**fn) * fvv * np.exp(-self.ks2*self.cs*self.css) + 0.25 * (
                        Fvaupi*((self.css-qq1)**fn) + Fvadni*((self.css+qq1)**fn)
                        + Fvaups*((self.cs+qq2)**fn) + Fvadns*((self.cs-qq2)**fn)
                        + Fvbupi*((self.css-qq5)**fn) + Fvbdni*((self.css+qq5)**fn)
                        + Fvbups*((self.cs+qq6)**fn) + Fvbdns*((self.cs-qq6)**fn))

            Ihh[n-1] = ((self.cs+self.css)**fn) * fhh * np.exp(-self.ks2*self.cs*self.css) + 0.25 * (
                        Fhaupi*((self.css-qq1)**fn) + Fhadni*((self.css+qq1)**fn)
                        + Fhaups*((self.cs+qq2)**fn) + Fhadns*((self.cs-qq2)**fn)
                        + Fhbupi*((self.css-qq5)**fn) + Fhbdni*((self.css+qq5)**fn)
                        + Fhbups*((self.cs+qq6)**fn) + Fhbdns*((self.cs-qq6)**fn))

            Ihv[n-1] = ((self.cs+self.css)**fn) * fhv * np.exp(-self.ks2*self.cs*self.css) + 0.25 * (
                        Fhvaupi*((self.css-qq1)**fn) + Fhvadni*((self.css+qq1)**fn)
                        + Fhvaups*((self.cs+qq2)**fn) + Fhvadns*((self.cs-qq2)**fn)
                        + Fhvbupi*((self.css-qq5)**fn) + Fhvbdni*((self.css+qq5)**fn)
                        + Fhvbups*((self.cs+qq6)**fn) + Fhvbdns*((self.cs-qq6)**fn))

            Ivh[n-1] = ((self.cs+self.css)**fn) * fvh * np.exp(-self.ks2*self.cs*self.css) + 0.25 * (
                        Fvhaupi*((self.css-qq1)**fn) + Fvhadni*((self.css+qq1)**fn)
                        + Fvhaups*((self.cs+qq2)**fn) + Fvhadns*((self.cs-qq2)**fn)
                        + Fvhbupi*((self.css-qq5)**fn) + Fvhbdni*((self.css+qq5)**fn)
                        + Fvhbups*((self.cs+qq6)**fn) + Fvhbdns*((self.cs-qq6)**fn))

            CIvv[n-1] = np.conj(Ivv[n-1])
            CIhh[n-1] = np.conj(Ihh[n-1])
            CIhv[n-1] = np.conj(Ihv[n-1])
            CIvh[n-1] = np.conj(Ivh[n-1])

        return Ivv, Ihh, Ihv, Ivh, CIvv, CIhh, CIhv, CIvh


    def compute_sigma0(self, Ivv, CIvv, Ihh, CIhh, Ihv, CIhv, Ivh, CIvh):
        
        sum1 = 0.
        sum2 = 0.
        sum3 = 0.
        sum4 = 0.
        temp = 1.

        for n in range(1, self.iterm):
            fn = n
            temp *= (self.ks2 / fn)
            sum1 += temp * (Ivv[n-1] * CIvv[n-1]) * self.spectra_1[n-1]
            sum2 += temp * (Ihh[n-1] * CIhh[n-1]) * self.spectra_1[n-1]
            sum3 += temp * (Ihv[n-1] * CIhv[n-1]) * self.spectra_1[n-1]
            sum4 += temp * (Ivh[n-1] * CIvh[n-1]) * self.spectra_1[n-1]

        allterm0 = 0.5 * np.exp(-self.ks2 * (self.cs2 + self.css2)) * sum1
        allterm1 = 0.5 * np.exp(-self.ks2 * (self.cs2 + self.css2)) * sum2
        allterm2 = 0.5 * np.exp(-self.ks2 * (self.cs2 + self.css2)) * sum3
        allterm3 = 0.5 * np.exp(-self.ks2 * (self.cs2 + self.css2)) * sum4

        sigma00 = np.real(allterm0)
        sigma01 = np.real(allterm1)
        sigma02 = np.real(allterm2)
        sigma03 = np.real(allterm3)

        VV = 10 * np.log10(sigma00)
        HH = 10 * np.log10(sigma01)
        HV = 10 * np.log10(sigma02)
        VH = 10 * np.log10(sigma03)

        return VV, HH, HV, VH


    def expal(self, q):
        return np.exp(-self.ks2 * (q**2.0 - q * (self.css - self.cs)))


    def _c1(self, zx, zy, zxp):
        return -self.csfs * (-1.0 - zx * zxp) + self.sfs * zxp * zy


    def _c2(self, zx, zy, zxp, zyp, u, v, q):
        return -self.csfs * (-self.cs * q - self.cs * u * zx - q * self.si * zxp - self.si * u * zx * zxp - self.cs * v * zyp - self.si * v * zx * zyp) \
            + self.sfs * (self.cs * u * zy + self.si * u * zxp * zy + q * self.si * zyp - self.cs * u * zyp + self.si * v * zy * zyp)


    def _c3(self, zx, zy, zxp, u, v, q):
        return -self.csfs * (self.si * u - q * self.si * zx - self.cs * u * zxp + self.cs * q * zx * zxp) \
            + self.sfs * (-self.si * v + self.cs * v * zxp + q * self.si * zy - self.cs * q * zxp * zy)


    def _c4(self, zx, zy, zxp, zyp):
        return -self.css * self.sfs * (-self.si * zyp + self.cs * zx * zyp) - self.csfs * self.css * (-self.cs - self.si * zxp - self.cs * zy * zyp) \
            + self.sis * (-self.cs * zx - self.si * zx * zxp - self.si * zy * zyp)


    def _c5(self, zx, zy, zxp, u, v, q):
        return -self.css * self.sfs * (-v * zx + v * zxp) - self.csfs * self.css * (q + u * zxp + v * zy) \
            + self.sis * (q * zx + u * zx * zxp + v * zxp * zy)


    def _c6(self, zx, zy, zyp, u, v, q):
        return -self.css * self.sfs * (-u * zyp + q * zx * zyp) - self.csfs * self.css * (v * zyp - q * zy * zyp) \
            + self.sis * (v * zx * zyp - u * zy * zyp)


    def _b1(self, zx, zy, zxp):
        return -self.css * self.sfs * (-1.0 - zx * zxp) - self.sis * zy - self.csfs * self.css * zxp * zy
    

    def _b2(self, zx, zy, zxp, zyp, u, v, q):
        return -self.css * self.sfs * (-self.cs * q - self.cs * u * zx - q * self.si * zxp - self.si * u * zx * zxp - self.cs * v * zyp - self.si * v * zx * zyp) \
            + self.sis * (-self.cs * q * zy - q * self.si * zxp * zy + q * self.si * zx * zyp - self.cs * u * zx * zyp - self.cs * v * zy * zyp) \
            - self.csfs * self.css * (self.cs * u * zy + self.si * u * zxp * zy + q * self.si * zyp - self.cs * u * zyp + self.si * v * zy * zyp)


    def _b3(self, zx, zy, zxp, u, v, q):
        return -self.css * self.sfs * (self.si * u - q * self.si * zx - self.cs * u * zxp + self.cs * q * zx * zxp) \
            - self.csfs * self.css * (-self.si * v + self.cs * v * zxp + q * self.si * zy - self.cs * q * zxp * zy) \
            + self.sis * (-self.si * v * zx + self.cs * v * zx * zxp + self.si * u * zy - self.cs * u * zxp * zy)


    def _b4(self, zx, zy, zxp, zyp):
        return -self.csfs * (-self.si * zyp + self.cs * zx * zyp) + self.sfs * (-self.cs - self.si * zxp - self.cs * zy * zyp)


    def _b5(self, zx, zy, zxp, u, v, q):
        return -self.csfs * (-v * zx + v * zxp) + self.sfs * (q + u * zxp + v * zy)


    def _b6(self, zx, zy, zyp, u, v, q):
        return -self.csfs * (-u * zyp + q * zx * zyp) + self.sfs * (v * zyp - q * zy * zyp)


    def cal_zp_fa(self, u, v, qslp):

        kxu = self.si + u
        ksxu = self.sis * self.csfs + u
        kyv = v
        ksyv = self.sis * self.sfs + v
        
        if abs(self.css - qslp.real) < self.ERROR:
            zx = 0.0
            zy = 0.0
        else:
            zx = -ksxu / (self.css - qslp)
            zy = -ksyv / (self.css - qslp)
            
        if abs((self.cs + qslp).real) < self.ERROR:
            zxp = 0.0
            zyp = 0.0
        else:
            zxp = kxu / (self.cs + qslp)
            zyp = kyv / (self.cs + qslp)
        
        return zx, zy, zxp, zyp


    def cal_zp_fb(self, u, v, qslp):

        kxu = self.si + u
        ksxu = self.sis * self.csfs + u
        kyv = v
        ksyv = self.sis * self.sfs + v

        zx = np.where(np.abs(np.real(self.css-qslp)) < self.ERROR, np.zeros_like(qslp), -ksxu/(self.css-qslp))
        zy = np.where(np.abs(np.real(self.css-qslp)) < self.ERROR, np.zeros_like(qslp), -ksyv/(self.css-qslp))

        zxp = np.where(np.abs(np.real(self.cs+qslp)) < self.ERROR, np.zeros_like(qslp), kxu/(self.cs+qslp))
        zyp = np.where(np.abs(np.real(self.cs+qslp)) < self.ERROR, np.zeros_like(qslp), kyv/(self.cs+qslp))

        return zx, zy, zxp, zyp


    def fahh(self, u, v, q, qslp, qfix, rh):

        zx, zy, zxp, zyp = self.cal_zp_fa(u, v, qslp)
        
        c1 = self._c1(zx, zy, zxp)
        c2 = self._c2(zx, zy, zxp, zyp, u, v, q)
        c3 = self._c3(zx, zy, zxp, u, v, q)
        c4 = self._c4(zx, zy, zxp, zyp)
        c5 = self._c5(zx, zy, zxp, u, v, q)
        c6 = self._c6(zx, zy, zyp, u, v, q)
        
        rph = 1.0 + rh
        rmh = 1.0 - rh
        ah = rph / qfix
        bh = rmh / qfix
        
        return -bh * (-rph * c1 + rmh * c2 + rph * c3) - ah * (rmh * c4 + rph * c5 + rmh * c6)


    def fahv(self, u, v, q, qslp, qfix, rhv):
        
        zx, zy, zxp, zyp = self.cal_zp_fa(u, v, qslp)
        
        b1 = self._b1(zx, zy, zxp)
        b2 = self._b2(zx, zy, zxp, zyp, u, v, q)
        b3 = self._b3(zx, zy, zxp, u, v, q)
        b4 = self._b4(zx, zy, zxp, zyp)
        b5 = self._b5(zx, zy, zxp, u, v, q)
        b6 = self._b6(zx, zy, zyp, u, v, q)
        
        rp = 1.0 + rhv
        rm = 1.0 - rhv
        a = rp / qfix
        b = rm / qfix
        
        return b * (rp * b1 - rm * b2 - rp * b3) + a * (rm * b4 + rp * b5 + rm * b6)


    def favh(self, u, v, q, qslp, qfix, rvh):

        zx, zy, zxp, zyp = self.cal_zp_fa(u, v, qslp)

        b1 = self._b1(zx, zy, zxp)
        b2 = self._b2(zx, zy, zxp, zyp, u, v, q)
        b3 = self._b3(zx, zy, zxp, u, v, q)
        b4 = self._b4(zx, zy, zxp, zyp)
        b5 = self._b5(zx, zy, zxp, u, v, q)
        b6 = self._b6(zx, zy, zyp, u, v, q)

        rp = 1.0 + rvh
        rm = 1.0 - rvh
        a = rp / qfix
        b = rm / qfix
        
        return b * (rp * b4 + rm * b5 + rp * b6) - a * (-rm * b1 + rp * b2 + rm * b3)


    def favv(self, u, v, q, qslp, qfix, rv):
        
        zx, zy, zxp, zyp = self.cal_zp_fa(u, v, qslp)
        
        c1 = self._c1(zx, zy, zxp)
        c2 = self._c2(zx, zy, zxp, zyp, u, v, q)
        c3 = self._c3(zx, zy, zxp, u, v, q)
        c4 = self._c4(zx, zy, zxp, zyp)
        c5 = self._c5(zx, zy, zxp, u, v, q)
        c6 = self._c6(zx, zy, zyp, u, v, q)
        
        rpv = 1.0 + rv
        rmv = 1.0 - rv
        av = rpv / qfix
        bv = rmv / qfix

        return bv * (-rpv * c1 + rmv * c2 + rpv * c3) + av * (rmv * c4 + rpv * c5 + rmv * c6)


    def fbhh(self, u, v, q, qslp, qfix, rh):
        
        zx, zy, zxp, zyp = self.cal_zp_fb(u, v, qslp)
        
        c1 = self._c1(zx, zy, zxp)
        c2 = self._c2(zx, zy, zxp, zyp, u, v, q)
        c3 = self._c3(zx, zy, zxp, u, v, q)
        c4 = self._c4(zx, zy, zxp, zyp)
        c5 = self._c5(zx, zy, zxp, u, v, q)
        c6 = self._c6(zx, zy, zyp, u, v, q)
        
        rph = 1.0 + rh
        rmh = 1.0 - rh
        ah = rph / qfix
        bh = rmh / qfix
        
        return ah * (-rph * c1 * self.er + rmh * c2 + rph * c3) + bh * (rmh * c4 + rph * c5 + rmh * c6 / self.er)


    def fbhv(self, u, v, q, qslp, qfix, rhv):

        zx, zy, zxp, zyp = self.cal_zp_fb(u, v, qslp)
        
        b1 = self._b1(zx, zy, zxp)
        b2 = self._b2(zx, zy, zxp, zyp, u, v, q)
        b3 = self._b3(zx, zy, zxp, u, v, q)
        b4 = self._b4(zx, zy, zxp, zyp)
        b5 = self._b5(zx, zy, zxp, u, v, q)
        b6 = self._b6(zx, zy, zyp, u, v, q)

        rp = 1.0 + rhv
        rm = 1.0 - rhv
        a = rp / qfix
        b = rm / qfix

        return a * (-rp * b1 + rm * b2 + rp * b3 / self.er) - b * (rm * b4 * self.er + rp * b5 + rm * b6)


    def fbvh(self, u, v, q, qslp, qfix, rvh):

        zx, zy, zxp, zyp = self.cal_zp_fb(u, v, qslp)
        
        b1 = self._b1(zx, zy, zxp)
        b2 = self._b2(zx, zy, zxp, zyp, u, v, q)
        b3 = self._b3(zx, zy, zxp, u, v, q)
        b4 = self._b4(zx, zy, zxp, zyp)
        b5 = self._b5(zx, zy, zxp, u, v, q)
        b6 = self._b6(zx, zy, zyp, u, v, q)
        
        rp = 1.0 + rvh
        rm = 1.0 - rvh
        a = rp / qfix
        b = rm / qfix
        
        return -a * (rp * b4 + rm * b5 + rp * b6 / self.er) + b * (-rm * b1 * self.er + rp * b2 + rm * b3)


    def fbvv(self, u, v, q, qslp, qfix, rv):

        zx, zy, zxp, zyp = self.cal_zp_fb(u, v, qslp)
        
        c1 = self._c1(zx, zy, zxp)
        c2 = self._c2(zx, zy, zxp, zyp, u, v, q)
        c3 = self._c3(zx, zy, zxp, u, v, q)
        c4 = self._c4(zx, zy, zxp, zyp)
        c5 = self._c5(zx, zy, zxp, u, v, q)
        c6 = self._c6(zx, zy, zyp, u, v, q)

        rpv = 1.0 + rv
        rmv = 1.0 - rv
        av = rpv / qfix
        bv = rmv / qfix
        
        return av * (rpv * c1 - rmv * c2 - rpv * c3 / self.er) - bv * (rmv * c4 * self.er + rpv * c5 + rmv * c6)