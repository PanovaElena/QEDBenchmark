#pragma once
#include "Constants.h"
#include "Ensemble.h"
#include "Grid.h"
#include "AnalyticalField.h"
#include "Pusher.h"
#include "Vectors.h"

#include "compton.h"
#include "breit_wheeler.h"
#include "macros.h"

#include <stdint.h>
#include <omp.h>
#include <random>

using namespace constants;
namespace pfc
{
    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED : public ParticlePusher
    {
        // Optimized Vector for QED
        template <class T> class MyQEDVector;

    public:

        Scalar_Fast_QED() : compton(), breit_wheeler(),
            schwingerField(sqr(Constants<FP>::electronMass()* Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck()))
        {
            const int maxThreads = OMP_GET_MAX_THREADS();

            this->randGenerator.resize(maxThreads);
            this->distribution.resize(maxThreads);
            for (int thr = 0; thr < maxThreads; thr++) {
                this->randGenerator[thr].seed();
                this->distribution[thr] = std::uniform_real_distribution<FP>(0.0, 1.0);
            }
        }

        void disablePhotonEmission()
        {
            this->photonEmissionEnabled = false;
        }

        void enablePhotonEmission()
        {
            this->photonEmissionEnabled = true;
        }

        void disablePairProduction()
        {
            this->pairProductionEnabled = false;
        }

        void enablePairProduction()
        {
            this->pairProductionEnabled = true;
        }

        // only for benchmark
        void processParticles(Ensemble3d* particles, const std::vector<std::vector<FP3>>& e,
            const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<MyQEDVector<Ensemble3d::ParticleType>> generatedParticles(maxThreads);
            std::vector<MyQEDVector<Ensemble3d::ParticleType>> generatedPhotons(maxThreads);

            if ((*particles)[Photon].size())
                handleParticleArray((*particles)[ParticleTypes::Photon], e[0], b[0], timeStep,
                    generatedParticles, generatedPhotons);
            if ((*particles)[Electron].size())
                handleParticleArray((*particles)[ParticleTypes::Electron], e[1], b[1], timeStep,
                    generatedParticles, generatedPhotons);
            if ((*particles)[Positron].size())
                handleParticleArray((*particles)[ParticleTypes::Positron], e[2], b[2], timeStep,
                    generatedParticles, generatedPhotons);

            // add generated particles to particleArray
            for (int th = 0; th < maxThreads; th++)
            {
                for (int ind = 0; ind < generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(generatedParticles[th][ind]);
                }
            }
        }

        void handleParticleArray(Ensemble3d::ParticleArray& particles,
            const std::vector<FP3>& e,
            const std::vector<FP3>& b, FP timeStep,
            std::vector<MyQEDVector<Ensemble3d::ParticleType>>& generatedParticles,
            std::vector<MyQEDVector<Ensemble3d::ParticleType>>& generatedPhotons)
        {
            const int n = particles.size();
            const int nChunks = (n + chunkSize - 1) / chunkSize;

#pragma omp parallel for
            for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++)
            {
                int threadId = OMP_GET_THREAD_NUM();

                const int begin = chunkIndex * chunkSize,
                    end = std::min(begin + chunkSize, n), size = end - begin;

                if (particles.getType() == ParticleTypes::Photon)
                {
                    handlePhotonChunk(particles, begin, end,
                        e.data() + begin, b.data() + begin, timeStep,
                        generatedParticles[threadId], generatedPhotons[threadId]);
                }
                else if (particles.getType() == ParticleTypes::Electron ||
                    particles.getType() == ParticleTypes::Positron)
                {
                    handleParticleChunk(particles, begin, end,
                        e.data() + begin, b.data() + begin, timeStep,
                        generatedParticles[threadId], generatedPhotons[threadId]);
                }
            }

            // remove old photons
            if (particles.getType() == ParticleTypes::Photon)
                particles.clear();
        }

        // main multithreaded interface
        void processParticles(Ensemble3d* particles, TGrid* grid, FP timeStep)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<MyQEDVector<Ensemble3d::ParticleType>> generatedParticles(maxThreads);
            std::vector<MyQEDVector<Ensemble3d::ParticleType>> generatedPhotons(maxThreads);

            if ((*particles)[Photon].size())
                handleParticleArray((*particles)[ParticleTypes::Photon], grid, timeStep,
                    generatedParticles, generatedPhotons);
            if ((*particles)[Electron].size())
                handleParticleArray((*particles)[ParticleTypes::Electron], grid, timeStep,
                    generatedParticles, generatedPhotons);
            if ((*particles)[Positron].size())
                handleParticleArray((*particles)[ParticleTypes::Positron], grid, timeStep,
                    generatedParticles, generatedPhotons);

            // add generated particles to particleArray
            for (int th = 0; th < maxThreads; th++)
            {
                for (int ind = 0; ind < generatedPhotons[th].size(); ind++)
                {
                    particles->addParticle(generatedPhotons[th][ind]);
                }
                for (int ind = 0; ind < generatedParticles[th].size(); ind++)
                {
                    particles->addParticle(generatedParticles[th][ind]);
                }
            }
        }

        void handleParticleArray(Ensemble3d::ParticleArray& particles, TGrid* grid, FP timeStep,
            std::vector<MyQEDVector<Ensemble3d::ParticleType>>& generatedParticles,
            std::vector<MyQEDVector<Ensemble3d::ParticleType>>& generatedPhotons)
        {
            const int n = particles.size();
            const int nChunks = (n + chunkSize - 1) / chunkSize;

#pragma omp parallel for
            for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++)
            {
                int threadId = OMP_GET_THREAD_NUM();

                const int begin = chunkIndex * chunkSize,
                    end = std::min(begin + chunkSize, n), size = end - begin;

                FP3 eChunk[chunkSize], bChunk[chunkSize];
#pragma omp simd
                for (int i = 0; i < size; i++)
                {
                    eChunk[i] = grid->getE(particles[i + begin].getPosition());
                    bChunk[i] = grid->getB(particles[i + begin].getPosition());
                }

                if (particles.getType() == ParticleTypes::Photon)
                {
                    handlePhotonChunk(particles, begin, end, eChunk, bChunk, timeStep,
                        generatedParticles[threadId], generatedPhotons[threadId]);
                }
                else if (particles.getType() == ParticleTypes::Electron ||
                    particles.getType() == ParticleTypes::Positron)
                {
                    handleParticleChunk(particles, begin, end, eChunk, bChunk, timeStep,
                        generatedParticles[threadId], generatedPhotons[threadId]);
                }
            }

            // remove old photons
            if (particles.getType() == ParticleTypes::Photon)
                particles.clear();
        }

        forceinline void handlePhotonChunk(Ensemble3d::ParticleArray& photons,
            const int chunkBegin, const int chunkEnd,
            const FP3* eChunk, const FP3* bChunk, FP timeStep,
            MyQEDVector<Ensemble3d::ParticleType>& generatedParticles,
            MyQEDVector<Ensemble3d::ParticleType>& generatedPhotons
        )
        {
            MyQEDVector<FP> timeAvalancheParticles, timeAvalanchePhotons;
            MyQEDVector<FP3> eAvalancheParticles, bAvalancheParticles,
                eAvalanchePhotons, bAvalanchePhotons;
            MyQEDVector<Ensemble3d::ParticleType> avalancheParticles, avalanchePhotons;

            // start avalanche with current photons
            for (int i = chunkBegin; i < chunkEnd; i++) {
                avalanchePhotons.push_back(static_cast<Ensemble3d::ParticleType>(photons[i]));
                timeAvalanchePhotons.push_back((FP)0.0);
                eAvalanchePhotons.push_back(eChunk[i - chunkBegin]);
                bAvalanchePhotons.push_back(bChunk[i - chunkBegin]);
            }

            runAvalanche(timeStep,
                avalancheParticles, timeAvalancheParticles,
                eAvalancheParticles, bAvalancheParticles,
                avalanchePhotons, timeAvalanchePhotons,
                eAvalanchePhotons, bAvalanchePhotons);

            // pop non-physical photons
            removeNonPhysicalPhotons(avalanchePhotons);

            for (int i = 0; i < avalancheParticles.size(); i++)
                generatedParticles.push_back(avalancheParticles[i]);
            for (int i = 0; i < avalanchePhotons.size(); i++)
                generatedPhotons.push_back(avalanchePhotons[i]);
        }

        forceinline void handleParticleChunk(Ensemble3d::ParticleArray& particles,
            const int chunkBegin, const int chunkEnd,
            const FP3* eChunk, const FP3* bChunk, FP timeStep,
            MyQEDVector<Ensemble3d::ParticleType>& generatedParticles,
            MyQEDVector<Ensemble3d::ParticleType>& generatedPhotons
        )
        {
            MyQEDVector<FP> timeAvalancheParticles, timeAvalanchePhotons;
            MyQEDVector<FP3> eAvalancheParticles, bAvalancheParticles,
                eAvalanchePhotons, bAvalanchePhotons;
            MyQEDVector<Ensemble3d::ParticleType> avalancheParticles, avalanchePhotons;

            // start avalanche with current particles
            for (int i = chunkBegin; i < chunkEnd; i++) {
                avalancheParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);
                eAvalancheParticles.push_back(eChunk[i - chunkBegin]);
                bAvalancheParticles.push_back(bChunk[i - chunkBegin]);
            }

            runAvalanche(timeStep,
                avalancheParticles, timeAvalancheParticles,
                eAvalancheParticles, bAvalancheParticles,
                avalanchePhotons, timeAvalanchePhotons,
                eAvalanchePhotons, bAvalanchePhotons);

            // pop non-physical photons
            removeNonPhysicalPhotons(avalanchePhotons);
            // pop double particles
            rewriteParticlesInEnsemble(particles, avalancheParticles,
                chunkBegin, chunkEnd - chunkBegin);

            for (int i = 0; i < avalancheParticles.size(); i++)
                generatedParticles.push_back(avalancheParticles[i]);
            for (int i = 0; i < avalanchePhotons.size(); i++)
                generatedPhotons.push_back(avalanchePhotons[i]);
        }

        
        void runAvalanche(FP timeStep,
            MyQEDVector<Ensemble3d::ParticleType>& generatedParticles,
            MyQEDVector<FP>& timeAvalancheParticles,
            MyQEDVector<FP3>& eAvalancheParticles,
            MyQEDVector<FP3>& bAvalancheParticles,
            MyQEDVector<Ensemble3d::ParticleType>& generatedPhotons,
            MyQEDVector<FP>& timeAvalanchePhotons,
            MyQEDVector<FP3>& eAvalanchePhotons,
            MyQEDVector<FP3>& bAvalanchePhotons)
        {
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            while (countProcessedParticles != generatedParticles.size()
                || countProcessedPhotons != generatedPhotons.size())
            {
                int n = generatedParticles.size() - countProcessedParticles;
                int nChunks = (n + chunkSize - 1) / chunkSize;

                for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++)
                {
                    int begin = countProcessedParticles + chunkIndex * chunkSize,
                        end = std::min(begin + chunkSize, countProcessedParticles + n);

                    oneParticleStep(timeStep,
                        generatedParticles, timeAvalancheParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        generatedPhotons, timeAvalanchePhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        begin, end
                    );
                }
                countProcessedParticles = generatedParticles.size();

                n = generatedPhotons.size() - countProcessedPhotons;
                nChunks = (n + chunkSize - 1) / chunkSize;

                for (int chunkIndex = 0; chunkIndex < nChunks; chunkIndex++)
                {
                    int begin = countProcessedPhotons + chunkIndex * chunkSize,
                        end = std::min(begin + chunkSize, countProcessedPhotons + n);

                    onePhotonStep(timeStep,
                        generatedPhotons, timeAvalanchePhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        generatedParticles, timeAvalancheParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        begin, end
                    );
                }
                countProcessedPhotons = generatedPhotons.size();
            }
        }

        void oneParticleStep(FP timeStep,
            MyQEDVector<Ensemble3d::ParticleType>& particles,
            MyQEDVector<FP>& time, MyQEDVector<FP3>& e, MyQEDVector<FP3>& b,
            MyQEDVector<Ensemble3d::ParticleType>& generatedPhotons,
            MyQEDVector<FP>& timeAvalanchePhotons,
            MyQEDVector<FP3>& eAvalanchePhotons, MyQEDVector<FP3>& bAvalanchePhotons,
            int begin, int end)
        {
            begin = sortHandledParticles(particles, timeStep,
                time, e, b, begin, end);

            while (begin < end) {

                FP chi[chunkSize], dt[chunkSize], gamma[chunkSize], rate[chunkSize],
                    chi_new[chunkSize], delta[chunkSize], randomNumber[chunkSize];
                Ensemble3d::ParticleType newPhoton[chunkSize];

                int s = begin;
                int size = end - begin;

                // compute dt
#pragma omp simd
                for (int i = 0; i < size; i++) {
                    FP3 v = particles[s + i].getVelocity();
                    FP H_eff = sqr(e[s + i] + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b[s + i]))
                        - sqr(SP(e[s + i], v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    gamma[i] = particles[s + i].getGamma();
                    chi[i] = gamma[i] * H_eff / this->schwingerField;
                }

                for (int i = 0; i < size; i++)
                    if (chi[i] > 0.0 && this->photonEmissionEnabled)
                        randomNumber[i] = randomNumberOmp();

#pragma omp simd
                for (int i = 0; i < size; i++) {
                    rate[i] = (FP)0.0;
                    dt[i] = (FP)2 * timeStep;
                    if (chi[i] > 0.0 && this->photonEmissionEnabled)
                    {
                        rate[i] = compton.rate(chi[i]);
                        dt[i] = getDt(rate[i], chi[i], gamma[i], randomNumber[i]);
                    }
                }

                // move particles
#pragma omp simd
                for (int i = 0; i < size; i++) {
                    bool cond = dt[i] + time[s + i] > timeStep;
                    FP realDt = cond ? timeStep - time[s + i] : dt[i];
                    boris(particles[s + i], e[s + i], b[s + i], realDt);
                    time[s + i] = cond ? timeStep : time[s + i] + dt[i];
                }

                // eliminate finished particles
                int newBegin = sortHandledParticles(particles, timeStep, time, e, b,
                    begin, end, chi);
                int elimShift = newBegin - begin;
                begin = newBegin;
                s = begin;
                size = end - begin;

                // process other particles
#pragma ivdep
                for (int i = 0; i < size; i++)  // NOT VECTORIZED: func call (random number)
                    randomNumber[i] = randomNumberOmp();

#pragma omp simd  // omp simd because of inner unrolled loop
                for (int i = 0; i < size; i++) {
                    FP3 v = particles[s + i].getVelocity();
                    FP H_eff = sqr(e[s + i] + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b[s + i]))
                        - sqr(SP(e[s + i], v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particles[s + i].getGamma();
                    chi_new[i] = gamma * H_eff / this->schwingerField;
                }

#pragma omp simd
                for (int i = 0; i < size; i++) {
                    delta[i] = photonGenerator((chi[i + elimShift] + chi_new[i]) / (FP)2.0, randomNumber[i]);
                }

#pragma omp simd
                for (int i = 0; i < size; i++) {
                    newPhoton[i].setType(Photon);
                    newPhoton[i].setWeight(particles[s + i].getWeight());
                    newPhoton[i].setPosition(particles[s + i].getPosition());
                    newPhoton[i].setMomentum(delta[i] * particles[s + i].getMomentum());
                }

                generatedPhotons.extend(newPhoton, size);
                timeAvalanchePhotons.extend(time.data() + s, size);
                eAvalanchePhotons.extend(e.data() + s, size);
                bAvalanchePhotons.extend(b.data() + s, size);

#pragma omp simd
                for (int i = 0; i < size; i++) {
                    particles[s + i].setMomentum(((FP)1 - delta[i]) * particles[s + i].getMomentum());
                }
            }
        }

        void onePhotonStep(FP timeStep,
            MyQEDVector<Ensemble3d::ParticleType>& photons,
            MyQEDVector<FP>& time, MyQEDVector<FP3>& e, MyQEDVector<FP3>& b,
            MyQEDVector<Ensemble3d::ParticleType>& generatedParticles,
            MyQEDVector<FP>& timeAvalancheParticles,
            MyQEDVector<FP3>& eAvalancheParticles, MyQEDVector<FP3>& bAvalancheParticles,
            int begin, int end)
        {
            int size = end - begin;

            FP3 k_[chunkSize];
            FP chi[chunkSize], dt[chunkSize], gamma[chunkSize], rate[chunkSize],
                delta[chunkSize], randomNumber[chunkSize];
            Ensemble3d::ParticleType newParticle[chunkSize];

            int s = begin;

            // compute dt
#pragma omp simd
            for (int i = 0; i < size; i++) {
                k_[i] = photons[s + i].getVelocity();
                k_[i] = ((FP)1 / k_[i].norm()) * k_[i]; // normalized wave vector
                FP H_eff = sqrt(sqr(e[s + i] + VP(k_[i], b[s + i])) - sqr(SP(e[s + i], k_[i])));
                gamma[i] = photons[s + i].getMomentum().norm()
                    / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
                chi[i] = gamma[i] * H_eff / this->schwingerField;
            }

            for (int i = 0; i < size; i++)
                if (chi[i] > 0.0 && this->pairProductionEnabled)
                    randomNumber[i] = randomNumberOmp();

#pragma omp simd
            for (int i = 0; i < size; i++) {
                rate[i] = 0.0;
                dt[i] = (FP)2 * timeStep;
                if (chi[i] > 0.0 && this->pairProductionEnabled)
                {
                    rate[i] = breit_wheeler.rate(chi[i]);
                    dt[i] = getDt(rate[i], chi[i], gamma[i], randomNumber[i]);
                }
            }

            // move photons
#pragma omp simd
            for (int i = 0; i < size; i++) {
                bool cond = dt[i] + time[s + i] > timeStep;
                FP realDt = cond ? timeStep - time[s + i] : dt[i];
                photons[s + i].setPosition(photons[s + i].getPosition()
                    + realDt * Constants<FP>::lightVelocity() * k_[i]);
                time[s + i] = cond ? timeStep : time[s + i] + dt[i];
            }

            // eliminate finished photons
            int newBegin = sortHandledParticles(photons, timeStep, time, e, b,
                begin, end, chi);
            int elimShift = newBegin - begin;
            size -= elimShift;
            begin = newBegin;
            s = begin;

            // process other particles
            for (int i = 0; i < size; i++)
                randomNumber[i] = randomNumberOmp();

#pragma omp simd
            for (int i = 0; i < size; i++) {
                delta[i] = pairGenerator(chi[i + elimShift], randomNumber[i]);
            }

#pragma omp simd
            for (int i = 0; i < size; i++) {
                newParticle[i].setType(Electron);
                newParticle[i].setWeight(photons[s + i].getWeight());
                newParticle[i].setPosition(photons[s + i].getPosition());
                newParticle[i].setMomentum(delta[i] * photons[s + i].getMomentum());
            }

            generatedParticles.extend(newParticle, size);
            timeAvalancheParticles.extend(time.data() + s, size);
            eAvalancheParticles.extend(e.data() + s, size);
            bAvalancheParticles.extend(b.data() + s, size);

#pragma omp simd
            for (int i = 0; i < size; i++) {
                newParticle[i].setType(Positron);
                newParticle[i].setMomentum(((FP)1 - delta[i]) * photons[s + i].getMomentum());
            }

            generatedParticles.extend(newParticle, size);
            timeAvalancheParticles.extend(time.data() + s, size);
            eAvalancheParticles.extend(e.data() + s, size);
            bAvalancheParticles.extend(b.data() + s, size);

#pragma omp simd
            for (int i = 0; i < size; i++) {
                time[s + i] = timeStep;
            }

#pragma omp simd
            for (int i = 0; i < size; i++) {
                // mark old photon as deleted
                photons[s + i].setP(Particle3d::MomentumType());
            }
        }

        // rewrite 2 borises
        forceinline void boris(Ensemble3d::ParticleType&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        forceinline void boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
        {
            FP eCoeff = timeStep * particle.getCharge() / ((FP)2 * particle.getMass() * Constants<FP>::lightVelocity());
            FP3 eMomentum = e * eCoeff;
            FP3 um = particle.getP() + eMomentum;
            FP3 t = b * eCoeff / sqrt((FP)1 + um.norm2());
            FP3 uprime = um + cross(um, t);
            FP3 s = t * (FP)2 / ((FP)1 + t.norm2());
            particle.setP(eMomentum + um + cross(uprime, s));
            particle.setPosition(particle.getPosition() + timeStep * particle.getVelocity());
        }

        forceinline int sortHandledParticles(MyQEDVector<Ensemble3d::ParticleType>& particles,
            FP timeStep, MyQEDVector<FP>& time, MyQEDVector<FP3>& e, MyQEDVector<FP3>& b,
            int begin, int end, FP* const chi = nullptr)
        {
            int lastIndex = begin;
            for (int i = begin; i < end; i++)
                if (time[i] >= timeStep) {
                    if (lastIndex != i) {
                        std::swap(particles[i], particles[lastIndex]);
                        std::swap(time[i], time[lastIndex]);
                        std::swap(e[i], e[lastIndex]);
                        std::swap(b[i], b[lastIndex]);
                        if (chi) std::swap(chi[i - begin], chi[lastIndex - begin]);
                    }
                    lastIndex++;
                }
            return lastIndex;
        }

        forceinline void removeNonPhysicalPhotons(
            MyQEDVector<Ensemble3d::ParticleType>& photons)
        {
            int lastIndex = 0;
            for (int k = 0; k < photons.size(); k++)
                if (photons[k].getGamma() != (FP)1.0) {
                    if (lastIndex != k)
                        std::swap(photons[k], photons[lastIndex]);
                    lastIndex++;
                }
            photons.decrease(lastIndex);
        }

        forceinline void rewriteParticlesInEnsemble(
            Ensemble3d::ParticleArray& particles,
            MyQEDVector<Ensemble3d::ParticleType>& avalancheParticles,
            int beginParticles, int size)
        {
            // replace old particles with new particles
            for (int i = 0; i < size; i++)
                particles[beginParticles + i] = avalancheParticles[i];
            // pop double particles
            int nMove = avalancheParticles.size() - size <= size ?
                avalancheParticles.size() - size : size;
            for (int j = 0; j < nMove; j++)
                std::swap(avalancheParticles[j],
                    avalancheParticles[avalancheParticles.size() - nMove + j]);
            avalancheParticles.decrease(avalancheParticles.size() - size);
        }

        forceinline FP getDt(FP rate, FP chi, FP gamma, FP randomNumber)
        {
            FP r = -log(randomNumber);
            r *= gamma / chi;
            return r / rate;
        }

        forceinline FP photonGenerator(FP chi, FP randomNumber)
        {
            return compton.inv_cdf(randomNumber, chi);
        }

        forceinline FP pairGenerator(FP chi, FP randomNumber)
        {
            if (randomNumber < (FP)0.5)
                return breit_wheeler.inv_cdf(randomNumber, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - randomNumber, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Ensemble3d::ParticleType* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        forceinline FP randomNumberOmp()
        {
            const int threadId = OMP_GET_THREAD_NUM();
            return distribution[threadId](randGenerator[threadId]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        const FP schwingerField;

        bool photonEmissionEnabled = true, pairProductionEnabled = true;

        static const int chunkSize = int(__CHUNK_SIZE__);

        // Implementation of optimized Vector for QED
        template <class T>
        class MyQEDVector {
            static const int factor = 2;

            int n = 0, capacity = chunkSize;
            std::vector<T> elements;

        public:

            MyQEDVector() : elements(capacity) {}

            forceinline T& operator[](int i) { return elements[i]; }
            forceinline const T& operator[](int i) const { return elements[i]; }

            forceinline int& size() { return n; }
            forceinline void clear() { n = 0; }
            forceinline T* data() { return elements.data(); }

            forceinline void decrease(int newSize) { n = newSize; }

            forceinline void increaseCapacity(int newCapacity) {
                elements.resize(newCapacity);
                capacity = newCapacity;
            }

            forceinline void push_back(const T& value) {
                if (n == capacity) increaseCapacity(factor * capacity);
                elements[n++] = value;
            }
            forceinline void extend(const T* values, int size) {
                while (n + size >= capacity) increaseCapacity(factor * capacity);
#pragma omp simd
                for (int i = 0; i < size; i++)
                    elements[n + i] = values[i];
                n += size;
            }
            forceinline void pop_back() { n--; }
        };
    };

    typedef Scalar_Fast_QED<YeeGrid> Scalar_Fast_QED_Yee;
    typedef Scalar_Fast_QED<PSTDGrid> Scalar_Fast_QED_PSTD;
    typedef Scalar_Fast_QED<PSATDGrid> Scalar_Fast_QED_PSATD;
    typedef Scalar_Fast_QED<AnalyticalField> Scalar_Fast_QED_Analytical;
}
