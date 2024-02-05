#include "QED_AEG.h"
#include "Fast_QED.h"
#include "AnalyticalField.h"
#include "ParticleGenerator.h"
#include "Ensemble.h"
#include "ParticleBoundaryCondition.h"
#include "Thinning.h"
#include "Enums.h"

#include <functional>
#include <fstream>

// !!!!!!!!!!!!!
namespace pfc {
    std::vector<ParticleType> ParticleInfo::typesVector = { {constants::electronMass, constants::electronCharge},//electron
                                        {constants::electronMass, -constants::electronCharge},//positron
                                        {constants::protonMass, -constants::electronCharge},//proton
                                        {constants::electronMass, 0.0 } };//photon
    const ParticleType* ParticleInfo::types = &ParticleInfo::typesVector[0];
}

class Output {
public:

    std::string outputDir = "./Hichi_Fast_Rewritten/BasicOutput/data/";
    std::vector<FP> dataX, dataY;
    Int3 matrixSize;
    FP3 minCoord, maxCoord;

    Output(Int3 matrixSize, FP3 minCoord, FP3 maxCoord) :
        matrixSize(matrixSize), minCoord(minCoord), maxCoord(maxCoord) {
        dataX.resize(matrixSize.x);
        dataY.resize(matrixSize.y);
    }

    void fillDataDensity(ParticleArray3d& particles, CoordinateEnum axisLabel, std::vector<FP>& data) {
        std::fill(data.begin(), data.end(), 0.0);

        int axis0 = (int)axisLabel;
        int axis1 = (axis0 + 1) % 3;
        int axis2 = (axis0 + 2) % 3;

        const FP cellVol = ((maxCoord - minCoord) / (FP3)matrixSize).volume();
        FP weight = 1.0 / (cellVol * (FP)matrixSize[axis1] * (FP)matrixSize[axis2]);

        auto itStart = particles.begin(), itEnd = particles.end();

        for (auto it = itStart; it < itEnd; ++it) {
            FP3 pos = (*it).getPosition();
            FP factor = (*it).getWeight();

            int index = int((pos[axis0] - minCoord[axis0]) /
                (maxCoord[axis0] - minCoord[axis0]) * matrixSize[axis0]);
            if (index >= 0 && index < matrixSize[axis0])
                data[index] += factor * weight;
        }
    }

    void fillDataEnergy(ParticleArray3d& particles, CoordinateEnum axisLabel, std::vector<FP>& data) {
        std::fill(data.begin(), data.end(), 0.0);

        int axis0 = (int)axisLabel;
        int axis1 = (axis0 + 1) % 3;
        int axis2 = (axis0 + 2) % 3;

        const FP cellVol = ((maxCoord - minCoord) / (FP3)matrixSize).volume();
        FP weight = 1.0 / (cellVol * (FP)matrixSize[axis1] * (FP)matrixSize[axis2]);

        auto itStart = particles.begin(), itEnd = particles.end();

        for (auto it = itStart; it < itEnd; ++it) {
            FP3 pos = (*it).getPosition();
            FP e = (*it).getVelocity().norm2() * constants::electronMass;

            FP c2 = constants::lightVelocity * constants::lightVelocity;
            FP m2c4 = (*it).getMass() * (*it).getMass() * c2 * c2;
            FP energy = sqrt(m2c4 + (*it).getMomentum().norm2() * c2);
            FP factor = (*it).getWeight();

            int index = int((pos[axis0] - minCoord[axis0]) /
                (maxCoord[axis0] - minCoord[axis0]) * matrixSize[axis0]);
            if (index >= 0 && index < matrixSize[axis0])
                data[index] += factor * weight * energy;
        }
    }

    void fillDataField(std::function<FP(FP, FP, FP, FP)> field, FP t,
        CoordinateEnum axisLabel, std::vector<FP>& data) {
        std::fill(data.begin(), data.end(), 0.0);

        int axis0 = (int)axisLabel;
        int axis1 = (axis0 + 1) % 3;
        int axis2 = (axis0 + 2) % 3;

        FP coord1 = (maxCoord[axis1] + minCoord[axis1]) * 0.5;
        FP coord2 = (maxCoord[axis2] + minCoord[axis2]) * 0.5;
        FP d = (maxCoord[axis0] - minCoord[axis0]) / data.size();

        for (int i = 0; i < data.size(); i++) {
            FP coord0 = minCoord[axis0] + i * d;
            data[i] = field(coord0, coord1, coord2, t);
        }
    }

    std::string generateFileName(int iter) {
        std::string fileName = std::to_string(iter);
        int len = fileName.size();
        for (int i = 0; i < 6 - len; i++)
            fileName = "0" + fileName;
        return fileName + ".txt";
    }

    void writeToFile(const std::vector<FP>& data, std::string dir, std::string fileName) {
        std::ofstream file(dir + "/" + fileName);

        for (int i = 0; i < data.size(); i++)
            file << data[i] << " ";

        file.close();
    }

    void run(int iter, FP time, Ensemble3d& ensemble, AnalyticalField& field) {
        std::string fileName = generateFileName(iter);

        fillDataDensity(ensemble[ParticleTypes::Electron], CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/Electron1Dx/", fileName);
        fillDataDensity(ensemble[ParticleTypes::Electron], CoordinateEnum::y, dataY);
        writeToFile(dataY, outputDir + "/Electron1Dy/", fileName);
        fillDataDensity(ensemble[ParticleTypes::Positron], CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/Positron1Dx/", fileName);
        fillDataDensity(ensemble[ParticleTypes::Photon], CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/Photon1Dx/", fileName);

        fillDataField(field.getEy(), time, CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/Ey1D/", fileName);
        fillDataField(field.getEz(), time, CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/Ez1D/", fileName);
        fillDataField(field.getBy(), time, CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/By1D/", fileName);
        fillDataField(field.getBz(), time, CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/Bz1D/", fileName);

        fillDataEnergy(ensemble[ParticleTypes::Photon], CoordinateEnum::x, dataX);
        writeToFile(dataX, outputDir + "/PhotonEnergy1Dx/", fileName);
    }

};


void removeNonPhysicalPhotons(Ensemble3d& ensemble, FP threshold) {
    ParticleArray3d& particles = ensemble[ParticleTypes::Photon];
    int nThreads = OMP_GET_MAX_THREADS();

    std::vector<std::vector<int>> deletedParticles(nThreads);

OMP_FOR()
    for (int i = 0; i < particles.size(); i++)
        if (particles[i].getMomentum().norm() * constants::c < threshold)
            deletedParticles[OMP_GET_THREAD_NUM()].push_back(i);

    for (int thr = nThreads - 1; thr >= 0; thr--)
        for (int i = (int)deletedParticles[thr].size() - 1; i >= 0; i--)
            particles.deleteParticle(deletedParticles[thr][i]);
}

void removePhotonsOutOfDomain(Ensemble3d& ensemble, const FP3& minCoords, const FP3& maxCoords) {
    ParticleArray3d& particles = ensemble[ParticleTypes::Photon];
    int nThreads = OMP_GET_MAX_THREADS();

    std::vector<std::vector<int>> deletedParticles(nThreads);

OMP_FOR()
    for (int i = 0; i < particles.size(); i++)
        if (!(particles[i].getPosition() >= minCoords && particles[i].getPosition() < maxCoords))
            deletedParticles[OMP_GET_THREAD_NUM()].push_back(i);

    for (int thr = nThreads - 1; thr >= 0; thr--)
        for (int i = (int)deletedParticles[thr].size() - 1; i >= 0; i--)
            particles.deleteParticle(deletedParticles[thr][i]);
}

//void removeNonPhysicalPhotons(Ensemble3d& ensemble, FP threshold) {
//    ParticleArray3d& particles = ensemble[ParticleTypes::Photon];
//
//    for (auto it = particles.begin(); it != particles.end(); it++)
//        if ((*it).getMomentum().norm() * constants::c < threshold)
//            particles.deleteParticle(it);
//}


void startThinning(Ensemble3d& ensemble, std::vector<int> limits,
    Thinning<ParticleArray3d>& thinning) {
    for (int typeIndex = 0; typeIndex < sizeParticleTypes; typeIndex++)
        if (ensemble[typeIndex].size() >= limits[typeIndex])
            thinning.leveling(ensemble[typeIndex]);
}


int main() {
    //omp_set_num_threads(1);

    // -------- some constants --------

    const FP eV = 1.6e-12;
    const FP planckConstant = 1.0545718e-27;

    const int MatrixSize_X = 128;
    const int MatrixSize_Y = 4;
    const int MatrixSize_Z = 4;
    const FP Wavelength = 0.8e-4;
    const FP Omega = 2 * constants::pi * constants::lightVelocity / Wavelength;
    const FP Amp = 1100.0;
    const FP k = 2.0 * constants::pi / Wavelength;
    const int StepsPerPeriod = 200;
    const FP TimeStep = Wavelength / constants::lightVelocity / StepsPerPeriod;

    const FP RSize = 1.0 * Wavelength;

    const FP X_Min = -0.5 * RSize;
    const FP X_Max = 0.5 * RSize;
    const FP Y_Min = -0.5 * RSize;
    const FP Y_Max = 0.5 * RSize;
    const FP Z_Min = -0.5 * RSize;
    const FP Z_Max = 0.5 * RSize;

    const FP Size_X = X_Max - X_Min;
    const FP Size_Y = Y_Max - Y_Min;
    const FP Size_Z = Z_Max - Z_Min;

    const FP Step_X = Size_X / MatrixSize_X;
    const FP Step_Y = Size_Y / MatrixSize_Y;
    const FP Step_Z = Size_Z / MatrixSize_Z;
    const FP CellVol = Step_X * Step_Y * Step_Z;

    const int NumParticles = 250000;
    const FP PlasmaXMin = X_Min;
    const FP PlasmaXMax = X_Max;
    const FP PlasmaYMin = Y_Min;
    const FP PlasmaYMax = Y_Max;
    const FP PlasmaZMin = Z_Min;
    const FP PlasmaZMax = Z_Max;

    const FP PlasmaVolume = (PlasmaXMax - PlasmaXMin) * (PlasmaYMax - PlasmaYMin) * (PlasmaZMax - PlasmaZMin);
    const FP Density = 1.0;
    const FP ParticlesFactor = Density * PlasmaVolume / NumParticles;

    const FP RelField = -constants::electronMass * Omega * constants::lightVelocity /
        constants::electronCharge;

    const FP RemovePhotonsBelow = 2.0 * constants::electronMass * constants::c * constants::c;

    const int IterationsNumber = 60; // 5 * StepsPerPeriod + 1; // 
    const int BOIterationPass = StepsPerPeriod / 20;

    Int3 gridSize(MatrixSize_X, MatrixSize_Y, MatrixSize_Z);
    FP3 minCoords(PlasmaXMin, PlasmaYMin, PlasmaZMin);
    FP3 maxCoords(PlasmaXMax, PlasmaYMax, PlasmaZMax);
    FP3 gridStep(Step_X, Step_Y, Step_Z);

    // -------- analytical field --------

    std::function<FP(FP, FP, FP, FP)> ex = [Amp, RelField, Omega, k](FP x, FP y, FP z, FP t) {
        return 0.0; 
    };
    std::function<FP(FP, FP, FP, FP)> ey = [Amp, RelField, Omega, k](FP x, FP y, FP z, FP t) {
        return Amp * RelField * cos(k * x) * sin(Omega * t);
    };
    std::function<FP(FP, FP, FP, FP)> ez = [Amp, RelField, Omega, k](FP x, FP y, FP z, FP t) {
        return Amp * RelField * cos(k * x) * cos(Omega * t);
    };

    std::function<FP(FP, FP, FP, FP)> bx = [Amp, RelField, Omega, k](FP x, FP y, FP z, FP t) {
        return 0.0;
    };
    std::function<FP(FP, FP, FP, FP)> by = [Amp, RelField, Omega, k](FP x, FP y, FP z, FP t) {
        return -Amp * RelField * sin(k * x) * sin(Omega * t);
    };
    std::function<FP(FP, FP, FP, FP)> bz = [Amp, RelField, Omega, k](FP x, FP y, FP z, FP t) {
        return -Amp * RelField * sin(k * x) * cos(Omega * t);
    };

    AnalyticalField field(ex, ey, ez, bx, by, bz);

    // -------- particles --------

    Ensemble<ParticleArray3d> particleEnsemble;
    ParticleGenerator particleGenerator;

    auto block = [](FP x, FP xmin, FP xmax) {
        return (x >= xmin && x < xmax) ? 1.0 : 0.0;
    };

    std::function<FP(FP, FP, FP)> chargedParticleDensity = [Density, block,
        PlasmaXMin, PlasmaXMax, PlasmaYMin, PlasmaYMax, PlasmaZMin, PlasmaZMax](FP x, FP y, FP z) {
        return Density * block(x, PlasmaXMin, PlasmaXMax) *
            block(y, PlasmaYMin, PlasmaYMax) * block(z, PlasmaZMin, PlasmaZMax);
    };
    std::function<FP(FP, FP, FP)> photonDensity = [](FP x, FP y, FP z) { return 0.0; };

    std::function<FP(FP, FP, FP)> initialTemperature = [](FP x, FP y, FP z) { return 0.0; };
    std::function<FP3(FP, FP, FP)> initialMomentum = [](FP x, FP y, FP z) { return FP3(0.0, 0.0, 0.0); };

    // generate electrons
    particleGenerator(&particleEnsemble[ParticleTypes::Electron],
        gridSize, minCoords, gridStep,
        chargedParticleDensity, initialTemperature, initialMomentum,
        ParticlesFactor, ParticleTypes::Electron);
    // generate positrons
    particleGenerator(&particleEnsemble[ParticleTypes::Positron],
        gridSize, minCoords, gridStep,
        chargedParticleDensity, initialTemperature, initialMomentum,
        ParticlesFactor, ParticleTypes::Positron);
    // generate photons
    particleGenerator(&particleEnsemble[ParticleTypes::Photon],
        gridSize, minCoords, gridStep,
        photonDensity, initialTemperature, initialMomentum,
        ParticlesFactor, ParticleTypes::Photon);

    PeriodicalParticleBoundaryConditions paricleBC;

    Thinning<ParticleArray3d> thinning;  // leveling
    const int thinningLimit = 1000000;
    std::vector<int> thinningLimits(sizeParticleTypes, thinningLimit);

    //ScalarQED_AEG_Analytical qed;
    Scalar_Fast_QED_Analytical_v6 qed;

    // -------- output --------

    Output output(gridSize, minCoords, maxCoords);

    // -------- modeling --------
    double tAll = OMP_GET_WTIME();
    double tQED = 0.0, tThinning = 0.0, tRemove = 0.0, tParticleBC = 0.0, tRemoveOut = 0.0, tCompField = 0.0;

    for (int iter = 0; iter < IterationsNumber; iter++) {
        double tModule = 0.0;

        if (iter % BOIterationPass == 0)
            output.run(iter / BOIterationPass, field.getTime(), particleEnsemble, field);

        tModule = OMP_GET_WTIME();
        removeNonPhysicalPhotons(particleEnsemble, RemovePhotonsBelow);
        tRemove += OMP_GET_WTIME() - tModule;

        tModule = OMP_GET_WTIME();
        startThinning(particleEnsemble, thinningLimits, thinning);
        tThinning += OMP_GET_WTIME() - tModule;

        tModule = OMP_GET_WTIME();
        std::vector<std::vector<FP3>> e(3), b(3);

        e[0].resize(particleEnsemble[Photon].size());
        b[0].resize(particleEnsemble[Photon].size());
        e[1].resize(particleEnsemble[Electron].size());
        b[1].resize(particleEnsemble[Electron].size());
        e[2].resize(particleEnsemble[Positron].size());
        b[2].resize(particleEnsemble[Positron].size());

        ParticleArray3d& photons = particleEnsemble[Photon];
#pragma omp parallel for
        for (int i = 0; i < photons.size(); i++) {
            FP3 pPos = photons[i].getPosition();
            e[0][i] = field.getE(pPos);
            b[0][i] = field.getB(pPos);
        }
        ParticleArray3d& electrons = particleEnsemble[Electron];
#pragma omp parallel for
        for (int i = 0; i < electrons.size(); i++) {
            FP3 pPos = electrons[i].getPosition();
            e[1][i] = field.getE(pPos);
            b[1][i] = field.getB(pPos);
        }
        ParticleArray3d& positrons = particleEnsemble[Positron];
#pragma omp parallel for
        for (int i = 0; i < positrons.size(); i++) {
            FP3 pPos = positrons[i].getPosition();
            e[2][i] = field.getE(pPos);
            b[2][i] = field.getB(pPos);
        }

        tCompField += OMP_GET_WTIME() - tModule;

        tModule = OMP_GET_WTIME();
        qed.processParticles(&particleEnsemble, e, b, TimeStep);
        tQED += OMP_GET_WTIME() - tModule;

        tModule = OMP_GET_WTIME();
        paricleBC.updateParticlePosition(minCoords, maxCoords, &particleEnsemble[ParticleTypes::Electron]);
        paricleBC.updateParticlePosition(minCoords, maxCoords, &particleEnsemble[ParticleTypes::Positron]);
        paricleBC.updateParticlePosition(minCoords, maxCoords, &particleEnsemble[ParticleTypes::Photon]);
        tParticleBC += OMP_GET_WTIME() - tModule;

        tModule = OMP_GET_WTIME();
        removePhotonsOutOfDomain(particleEnsemble, minCoords, maxCoords);  // in case of nans
        tRemoveOut += OMP_GET_WTIME() - tModule;

        field.advanceTime(TimeStep);
    }

    std::cout << "Total time " << OMP_GET_WTIME() - tAll << std::endl;
    std::cout << "Remove time " << tRemove << ", average time " << tRemove / IterationsNumber << std::endl;
    std::cout << "Thinning time " << tThinning << ", average time " << tThinning /IterationsNumber << std::endl;
    std::cout << "Field time " << tCompField << ", average time " << tCompField /IterationsNumber << std::endl;
    std::cout << "QED time " << tQED << ", average time " << tQED /IterationsNumber << std::endl;
    std::cout << "ParticleBC time " << tParticleBC << ", average time " << tParticleBC / IterationsNumber << std::endl;
    std::cout << "RemoveOut time " << tRemoveOut << ", average time " << tRemoveOut / IterationsNumber << std::endl;
}
