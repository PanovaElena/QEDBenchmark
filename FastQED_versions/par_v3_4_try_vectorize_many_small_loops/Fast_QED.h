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
    public:

        static const int chunkSize = int(__CHUNK_SIZE__);

        template <class T>
        struct VectorOfGenElements {
            constexpr static int factor = 2;

            int n = 0, capacity = chunkSize;
            std::vector<T> data;
            T* raw = nullptr;

            VectorOfGenElements() : data(capacity), raw(data.data()) {}

            forceinline T& operator[](int i) { return raw[i]; }
            forceinline const T& operator[](int i) const { return raw[i]; }

            forceinline int& size() { return n; }
            forceinline void clear() { n = 0; }
            forceinline T* ptr() { return raw; }

            forceinline void resize() {
                data.resize(capacity);
                raw = data.data();
            }
            forceinline void reserve(int size) {  // size <= chunkSize
                if (n + size >= capacity) capacity *= factor;
                resize();
            }

            forceinline void push_back(const T& value) {
                if (n == capacity) {
                    capacity *= factor;
                    resize();
                }
                raw[n++] = value;
            }
            forceinline void push_back_without_reserve(const T& value) {
                raw[n++] = value;
            }

            forceinline void changeSize(int newSize) { n = newSize; }  // without resize!
            forceinline void pop_back() { n--; }

            forceinline void removeElement(int index) {
                if (index < n - 1)
                    std::swap(raw[index], raw[n - 1]);
                pop_back();
            }

        };

        Scalar_Fast_QED() : compton(), breit_wheeler(),
            schwingerField(sqr(Constants<FP>::electronMass()* Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck()))
        {
            int maxThreads = OMP_GET_MAX_THREADS();

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

        void processParticles(Ensemble3d* particles, const std::vector<std::vector<FP3>>& e,
            const std::vector<std::vector<FP3>>& b, FP timeStep)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<VectorOfGenElements<Particle3d>> generatedParticles(maxThreads);
            std::vector<VectorOfGenElements<Particle3d>> generatedPhotons(maxThreads);

            if ((*particles)[Photon].size())
                handlePhotons((*particles)[Photon], e[0], b[0], timeStep, generatedParticles, generatedPhotons);
            if ((*particles)[Electron].size())
                handleParticles((*particles)[Electron], e[1], b[1], timeStep, generatedParticles, generatedPhotons);
            if ((*particles)[Positron].size())
                handleParticles((*particles)[Positron], e[2], b[2], timeStep, generatedParticles, generatedPhotons);

            (*particles)[Photon].clear();

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

        // rewrite 2 borises
        forceinline void boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
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

        void handlePhotons(ParticleArray3d& photons,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            std::vector<VectorOfGenElements<Particle3d>>& generatedParticlesThread,
            std::vector<VectorOfGenElements<Particle3d>>& generatedPhotonsThread)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<VectorOfGenElements<FP>> timeAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP>> timeAvalanchePhotonsThread(maxThreads);

            std::vector<VectorOfGenElements<FP3>> eAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP3>> eAvalanchePhotonsThread(maxThreads);
            std::vector<VectorOfGenElements<FP3>> bAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP3>> bAvalanchePhotonsThread(maxThreads);

            const int n = photons.size();
            if (n == 0) return;
            const int nChunks = n / chunkSize;
            const int chunkRem = n - nChunks * chunkSize;

#pragma omp parallel for
            for (int chunkNum = 0; chunkNum < nChunks; chunkNum++)
            {
                int threadId = OMP_GET_THREAD_NUM();

                const int begin = chunkNum * chunkSize;
                const int end = begin + chunkSize;

                handlePhotonChunk(photons, ve, vb, timeStep,
                    generatedParticlesThread[threadId], generatedPhotonsThread[threadId],
                    timeAvalancheParticlesThread[threadId], timeAvalanchePhotonsThread[threadId],
                    eAvalancheParticlesThread[threadId], bAvalancheParticlesThread[threadId],
                    eAvalanchePhotonsThread[threadId], bAvalanchePhotonsThread[threadId],
                    begin, end, threadId
                );
            }
            if (chunkRem != 0)
                handlePhotonChunk(photons, ve, vb, timeStep,
                    generatedParticlesThread[0], generatedPhotonsThread[0],
                    timeAvalancheParticlesThread[0], timeAvalanchePhotonsThread[0],
                    eAvalancheParticlesThread[0], bAvalancheParticlesThread[0],
                    eAvalanchePhotonsThread[0], bAvalanchePhotonsThread[0],
                    n - chunkRem, n, 0
                );
        }

        void handleParticles(ParticleArray3d& particles,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            std::vector<VectorOfGenElements<Particle3d>>& generatedParticlesThread,
            std::vector<VectorOfGenElements<Particle3d>>& generatedPhotonsThread)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<VectorOfGenElements<FP>> timeAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP>> timeAvalanchePhotonsThread(maxThreads);

            std::vector<VectorOfGenElements<FP3>> eAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP3>> eAvalanchePhotonsThread(maxThreads);
            std::vector<VectorOfGenElements<FP3>> bAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP3>> bAvalanchePhotonsThread(maxThreads);

            const int n = particles.size();
            if (n == 0) return;
            const int nChunks = n / chunkSize;
            const int chunkRem = n - nChunks * chunkSize;

#pragma omp parallel for
            for (int chunkNum = 0; chunkNum < nChunks; chunkNum++)
            {
                int threadId = OMP_GET_THREAD_NUM();

                const int begin = chunkNum * chunkSize;
                const int end = begin + chunkSize;

                handleParticleChunk(particles, ve, vb, timeStep,
                    generatedParticlesThread[threadId], generatedPhotonsThread[threadId],
                    timeAvalancheParticlesThread[threadId], timeAvalanchePhotonsThread[threadId],
                    eAvalancheParticlesThread[threadId], bAvalancheParticlesThread[threadId],
                    eAvalanchePhotonsThread[threadId], bAvalanchePhotonsThread[threadId],
                    begin, end, threadId
                );
            }
            if (chunkRem != 0)
                handleParticleChunk(particles, ve, vb, timeStep,
                    generatedParticlesThread[0], generatedPhotonsThread[0],
                    timeAvalancheParticlesThread[0], timeAvalanchePhotonsThread[0],
                    eAvalancheParticlesThread[0], bAvalancheParticlesThread[0],
                    eAvalanchePhotonsThread[0], bAvalanchePhotonsThread[0],
                    n - chunkRem, n, 0
                );
        }

        forceinline void handlePhotonChunk(ParticleArray3d& photons,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            VectorOfGenElements<Particle3d>& generatedParticles,
            VectorOfGenElements<Particle3d>& generatedPhotons,
            VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<FP>& timeAvalanchePhotons,
            VectorOfGenElements<FP3>& eAvalancheParticles,
            VectorOfGenElements<FP3>& bAvalancheParticles,
            VectorOfGenElements<FP3>& eAvalanchePhotons,
            VectorOfGenElements<FP3>& bAvalanchePhotons,
            const int begin, const int end, const int threadId
        )
        {
            int prevGeneratedParticlesSize = generatedParticles.size();
            int prevGeneratedPhotonsSize = generatedPhotons.size();

            // start avalanche with current particles
            for (int i = begin; i < end; i++) {
                generatedPhotons.push_back(photons[i]);
                timeAvalanchePhotons.push_back((FP)0.0);
                eAvalanchePhotons.push_back(ve[i]);
                bAvalanchePhotons.push_back(vb[i]);
            }

            runAvalanche(ve, vb, timeStep,
                generatedParticles, timeAvalancheParticles,
                eAvalancheParticles, bAvalancheParticles,
                generatedPhotons, timeAvalanchePhotons,
                eAvalanchePhotons, bAvalanchePhotons,
                prevGeneratedParticlesSize, prevGeneratedPhotonsSize, threadId);

            // pop non-physical photons
            removeNonPhysicalPhotons(generatedPhotons, prevGeneratedPhotonsSize);

            timeAvalancheParticles.clear();
            timeAvalanchePhotons.clear();
            eAvalancheParticles.clear();
            bAvalancheParticles.clear();
            eAvalanchePhotons.clear();
            bAvalanchePhotons.clear();
        }

        forceinline void handleParticleChunk(ParticleArray3d& particles,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            VectorOfGenElements<Particle3d>& generatedParticles,
            VectorOfGenElements<Particle3d>& generatedPhotons,
            VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<FP>& timeAvalanchePhotons,
            VectorOfGenElements<FP3>& eAvalancheParticles,
            VectorOfGenElements<FP3>& bAvalancheParticles,
            VectorOfGenElements<FP3>& eAvalanchePhotons,
            VectorOfGenElements<FP3>& bAvalanchePhotons,
            const int begin, const int end, const int threadId
        )
        {
            int prevGeneratedParticlesSize = generatedParticles.size();
            int prevGeneratedPhotonsSize = generatedPhotons.size();

            // start avalanche with current particles
            for (int i = begin; i < end; i++) {
                generatedParticles.push_back(particles[i]);
                timeAvalancheParticles.push_back((FP)0.0);
                eAvalancheParticles.push_back(ve[i]);
                bAvalancheParticles.push_back(vb[i]);
            }

            runAvalanche(ve, vb, timeStep,
                generatedParticles, timeAvalancheParticles,
                eAvalancheParticles, bAvalancheParticles,
                generatedPhotons, timeAvalanchePhotons,
                eAvalanchePhotons, bAvalanchePhotons,
                prevGeneratedParticlesSize, prevGeneratedPhotonsSize, threadId);

            // pop non-physical photons
            removeNonPhysicalPhotons(generatedPhotons, prevGeneratedPhotonsSize);
            // pop double particles
            removeDoubleParticles(particles, generatedParticles,
                begin, prevGeneratedParticlesSize, end - begin);

            timeAvalancheParticles.clear();
            timeAvalanchePhotons.clear();
            eAvalancheParticles.clear();
            bAvalancheParticles.clear();
            eAvalanchePhotons.clear();
            bAvalanchePhotons.clear();
        }

        void runAvalanche(const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            VectorOfGenElements<Particle3d>& generatedParticles, VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<FP3>& eAvalancheParticles, VectorOfGenElements<FP3>& bAvalancheParticles,
            VectorOfGenElements<Particle3d>& generatedPhotons, VectorOfGenElements<FP>& timeAvalanchePhotons,
            VectorOfGenElements<FP3>& eAvalanchePhotons, VectorOfGenElements<FP3>& bAvalanchePhotons,
            int generatedParticlesStart, int generatedPhotonsStart, const int threadId)
        {
            int countProcessedParticles = generatedParticlesStart;
            int countProcessedPhotons = generatedPhotonsStart;

            while (countProcessedParticles != generatedParticles.size()
                || countProcessedPhotons != generatedPhotons.size())
            {
                int n = generatedParticles.size() - countProcessedParticles;
                int nChunks = n / chunkSize;
                int chunkRem = n - nChunks * chunkSize;

                for (int chunkNum = 0; chunkNum < nChunks; chunkNum++)
                    oneParticleStep(generatedParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        timeAvalancheParticles, timeStep,
                        generatedPhotons, timeAvalanchePhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        countProcessedParticles + chunkNum * chunkSize, chunkSize,
                        generatedParticlesStart, threadId
                    );
                if (chunkRem != 0)
                    oneParticleStep(generatedParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        timeAvalancheParticles, timeStep,
                        generatedPhotons, timeAvalanchePhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        countProcessedParticles + n - chunkRem, chunkRem,
                        generatedParticlesStart, threadId
                    );
                countProcessedParticles = generatedParticles.size();

                n = generatedPhotons.size() - countProcessedPhotons;
                nChunks = n / chunkSize;
                chunkRem = n - nChunks * chunkSize;

                for (int chunkNum = 0; chunkNum < nChunks; chunkNum++)
                    onePhotonStep(generatedPhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        timeAvalanchePhotons, timeStep,
                        generatedParticles, timeAvalancheParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        countProcessedPhotons + chunkNum * chunkSize, chunkSize,
                        generatedPhotonsStart, threadId
                    );
                if (chunkRem != 0)
                    onePhotonStep(generatedPhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        timeAvalanchePhotons, timeStep,
                        generatedParticles, timeAvalancheParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        countProcessedPhotons + n - chunkRem, chunkRem,
                        generatedPhotonsStart, threadId
                    );
                countProcessedPhotons = generatedPhotons.size();
            }
        }

        void oneParticleStep(VectorOfGenElements<Particle3d>& particles,
            VectorOfGenElements<FP3>& ve, VectorOfGenElements<FP3>& vb,
            VectorOfGenElements<FP>& vtime, FP timeStep,
            VectorOfGenElements<Particle3d>& generatedPhotons,
            VectorOfGenElements<FP>& timeAvalanchePhotons,
            VectorOfGenElements<FP3>& eAvalanchePhotons,
            VectorOfGenElements<FP3>& bAvalanchePhotons,
            int begin, int size, int auxShift, const int threadId)
        {
            const int end = begin + size;
            begin = sortHandledParticles(particles, timeStep, vtime, ve, vb, begin, end, auxShift);

            while (begin < end) {

                FP chi[chunkSize], dt[chunkSize], gamma[chunkSize], rate[chunkSize],
                    chi_new[chunkSize], delta[chunkSize], randomNumber[chunkSize];
                Particle3d newPhoton[chunkSize];

                int s = begin, sAux = begin - auxShift;
                size = end - begin;

                // compute dt
#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    FP3 v = particles[s + i].getVelocity();
                    FP H_eff = sqr(ve[sAux + i] + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, vb[sAux + i]))
                        - sqr(SP(ve[sAux + i], v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    gamma[i] = particles[s + i].getGamma();
                    chi[i] = gamma[i] * H_eff / this->schwingerField;
                }

#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    rate[i] = (FP)0.0;
                    if (chi[i] > 0.0 && this->photonEmissionEnabled)
                    {
                        rate[i] = compton.rate(chi[i]);
                    }
                }

#pragma ivdep
                for (int i = 0; i < size; i++)  // NOT VECTORIZED: func call (random number)
                    if (chi[i] > 0.0 && this->photonEmissionEnabled)
                        randomNumber[i] = randomNumberOmp(threadId);

#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    dt[i] = (FP)2 * timeStep;
                    if (chi[i] > 0.0 && this->photonEmissionEnabled)
                    {
                        dt[i] = getDt(rate[i], chi[i], gamma[i], randomNumber[i]);
                    }
                }

                // move particles
#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    bool cond = dt[i] + vtime[sAux + i] > timeStep;
                    FP realDt = cond ? timeStep - vtime[sAux + i] : dt[i];
                    boris(particles[s + i], ve[sAux + i], vb[sAux + i], realDt);
                    vtime[sAux + i] = cond ? timeStep : vtime[sAux + i] + dt[i];
                }

                // eliminate finished particles
                int newBegin = sortHandledParticles(particles, timeStep, vtime, ve, vb,
                    begin, end, auxShift, chi);
                int elimShift = newBegin - begin;
                begin = newBegin;
                s = begin;
                sAux = begin - auxShift;
                size = end - begin;

                // process other particles
#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    FP3 v = particles[s + i].getVelocity();
                    FP H_eff = sqr(ve[sAux + i] + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, vb[sAux + i]))
                        - sqr(SP(ve[sAux + i], v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particles[s + i].getGamma();
                    chi_new[i] = gamma * H_eff / this->schwingerField;
                }

#pragma ivdep
                for (int i = 0; i < size; i++)  // NOT VECTORIZED: func call (random number)
                    randomNumber[i] = randomNumberOmp(threadId);

#pragma omp simd
                for (int i = 0; i < size; i++)
                    delta[i] = photonGenerator((chi[i + elimShift] + chi_new[i]) / (FP)2.0, randomNumber[i]);

#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    newPhoton[i].setType(Photon);
                    newPhoton[i].setWeight(particles[s + i].getWeight());
                    newPhoton[i].setPosition(particles[s + i].getPosition());
                    newPhoton[i].setMomentum(delta[i] * particles[s + i].getMomentum());
                }

                generatedPhotons.reserve(size);
                timeAvalanchePhotons.reserve(size);
                eAvalanchePhotons.reserve(size);
                bAvalanchePhotons.reserve(size);
#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    generatedPhotons.push_back_without_reserve(newPhoton[i]);
                    timeAvalanchePhotons.push_back_without_reserve(vtime[sAux + i]);
                    eAvalanchePhotons.push_back_without_reserve(ve[sAux + i]);
                    bAvalanchePhotons.push_back_without_reserve(vb[sAux + i]);
                }

#pragma ivdep
#pragma vector always
                for (int i = 0; i < size; i++) {
                    particles[s + i].setMomentum(((FP)1 - delta[i]) * particles[s + i].getMomentum());
                }
            }
        }

        void onePhotonStep(VectorOfGenElements<Particle3d>& photons,
            VectorOfGenElements<FP3>& ve, VectorOfGenElements<FP3>& vb,
            VectorOfGenElements<FP>& vtime, FP timeStep,
            VectorOfGenElements<Particle3d>& generatedParticles,
            VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<FP3>& eAvalancheParticles,
            VectorOfGenElements<FP3>& bAvalancheParticles,
            int begin, int size, int auxShift, const int threadId)
        {
            FP3 k_[chunkSize];
            FP chi[chunkSize], dt[chunkSize], gamma[chunkSize], rate[chunkSize],
                delta[chunkSize], randomNumber[chunkSize];
            Particle3d newParticle[chunkSize];

            int s = begin, sAux = begin - auxShift;

            // compute dt
#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                k_[i] = photons[s + i].getVelocity();
                k_[i] = ((FP)1 / k_[i].norm()) * k_[i]; // normalized wave vector
                FP H_eff = sqrt(sqr(ve[sAux + i] + VP(k_[i], vb[sAux + i])) - sqr(SP(ve[sAux + i], k_[i])));
                gamma[i] = photons[s + i].getMomentum().norm()
                    / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
                chi[i] = gamma[i] * H_eff / this->schwingerField;
            }

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                rate[i] = 0.0;
                if (chi[i] > 0.0 && this->pairProductionEnabled)
                {
                    rate[i] = breit_wheeler.rate(chi[i]);
                }
            }

#pragma ivdep
            for (int i = 0; i < size; i++)  // NOT VECTORIZED: func call (random number)
                if (chi[i] > 0.0 && this->pairProductionEnabled)
                    randomNumber[i] = randomNumberOmp(threadId);

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                dt[i] = (FP)2 * timeStep;
                if (chi[i] > 0.0 && this->pairProductionEnabled)
                {
                    dt[i] = getDt(rate[i], chi[i], gamma[i], randomNumber[i]);
                }
            }

            // move photons
#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                bool cond = dt[i] + vtime[sAux + i] > timeStep;
                FP realDt = cond ? timeStep - vtime[sAux + i] : dt[i];
                photons[s + i].setPosition(photons[s + i].getPosition()
                    + realDt * Constants<FP>::lightVelocity() * k_[i]);
                vtime[sAux + i] = cond ? timeStep : vtime[sAux + i] + dt[i];
            }

            // eliminate finished photons
            int newBegin = sortHandledParticles(photons, timeStep, vtime, ve, vb,
                begin, begin + size, auxShift, chi);
            int elimShift = newBegin - begin;
            size -= elimShift;
            begin = newBegin;
            s = begin;
            sAux = begin - auxShift;

            // process other particles
#pragma ivdep
            for (int i = 0; i < size; i++) // NOT VECTORIZED: func call (random number)
                randomNumber[i] = randomNumberOmp(threadId);

#pragma omp simd
            for (int i = 0; i < size; i++) {
                delta[i] = pairGenerator(chi[i + elimShift], randomNumber[i]);  // unrolled inner loop
            }

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                newParticle[i].setType(Electron);
                newParticle[i].setWeight(photons[s + i].getWeight());
                newParticle[i].setPosition(photons[s + i].getPosition());
                newParticle[i].setMomentum(delta[i] * photons[s + i].getMomentum());
            }

            generatedParticles.reserve(2 * size);
            timeAvalancheParticles.reserve(2 * size);
            eAvalancheParticles.reserve(2 * size);
            bAvalancheParticles.reserve(2 * size);

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                generatedParticles.push_back_without_reserve(newParticle[i]);
                timeAvalancheParticles.push_back_without_reserve(vtime[sAux + i]);
                eAvalancheParticles.push_back_without_reserve(ve[sAux + i]);
                bAvalancheParticles.push_back_without_reserve(vb[sAux + i]);
            }

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                newParticle[i].setType(Positron);
                newParticle[i].setMomentum(((FP)1 - delta[i]) * photons[s + i].getMomentum());
            }

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                generatedParticles.push_back_without_reserve(newParticle[i]);
                timeAvalancheParticles.push_back_without_reserve(vtime[sAux + i]);
                eAvalancheParticles.push_back_without_reserve(ve[sAux + i]);
                bAvalancheParticles.push_back_without_reserve(vb[sAux + i]);
            }

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {  // NOT VECTORIZED: strange function call, unrolled
                // mark old photon as deleted
                photons[s + i].setP(Particle3d::MomentumType());  // there is a function call here because of zero arg
            }

#pragma ivdep
#pragma vector always
            for (int i = 0; i < size; i++) {
                vtime[sAux + i] = timeStep;
            }
        }

        forceinline int sortHandledParticles(VectorOfGenElements<Particle3d>& particles, FP timeStep,
            VectorOfGenElements<FP>& vtime, VectorOfGenElements<FP3>& ve, VectorOfGenElements<FP3>& vb,
            int begin, int end, int auxShift, FP* const chi = nullptr)
        {
            int lastIndex = begin;
            for (int i = begin; i < end; i++)
                if (vtime[i - auxShift] >= timeStep) {
                    if (lastIndex != i) {
                        std::swap(particles[i], particles[lastIndex]);
                        std::swap(vtime[i - auxShift], vtime[lastIndex - auxShift]);
                        std::swap(ve[i - auxShift], ve[lastIndex - auxShift]);
                        std::swap(vb[i - auxShift], vb[lastIndex - auxShift]);
                        if (chi) std::swap(chi[i - begin], chi[lastIndex - begin]);
                    }
                    lastIndex++;
                }
            return lastIndex;
        }

        forceinline void removeNonPhysicalPhotons(VectorOfGenElements<Particle3d>& generatedPhotons,
            int prevGeneratedPhotonsSize)
        {
            int lastIndex = prevGeneratedPhotonsSize;
            for (int k = prevGeneratedPhotonsSize; k < generatedPhotons.size(); k++)
                if (generatedPhotons[k].getGamma() != (FP)1.0) {
                    if (lastIndex != k)
                        std::swap(generatedPhotons[k], generatedPhotons[lastIndex]);
                    lastIndex++;
                }
            generatedPhotons.changeSize(lastIndex);
        }

        forceinline void removeDoubleParticles(ParticleArray3d& particles,
            VectorOfGenElements<Particle3d>& generatedParticles,
            int begin1, int begin2, int size)
        {
            // save new position and momentum of the current particles
            for (int i = 0; i < size; i++) {
                particles[begin1 + i].setMomentum(generatedParticles[begin2 + i].getMomentum());
                particles[begin1 + i].setPosition(generatedParticles[begin2 + i].getPosition());
            }
            // pop double particles
            int nMove = generatedParticles.size() - (begin2 + size) <= size ?
                generatedParticles.size() - (begin2 + size) : size;
            for (int j = 0; j < nMove; j++)
                std::swap(generatedParticles[j + begin2],
                    generatedParticles[generatedParticles.size() - nMove + j]);
            generatedParticles.changeSize(generatedParticles.size() - size);
        }

        //forceinline FP getDt(FP rate, FP chi, FP gamma)
        //{
        //    FP r = -log(randomNumberOmp());
        //    r *= gamma / chi;
        //    return r / rate;
        //}

        forceinline FP getDt(FP rate, FP chi, FP gamma, FP randomNumber)
        {
            FP r = -log(randomNumber);
            r *= gamma / chi;
            return r / rate;
        }

        //forceinline FP photonGenerator(FP chi)
        //{
        //    FP r = randomNumberOmp();
        //    return compton.inv_cdf(r, chi);
        //}

        forceinline FP photonGenerator(FP chi, FP randomNumber)
        {
            return compton.inv_cdf(randomNumber, chi);
        }

        //forceinline FP pairGenerator(FP chi)
        //{
        //    FP r = randomNumberOmp();
        //    if (r < (FP)0.5)
        //        return breit_wheeler.inv_cdf(r, chi);
        //    else
        //        return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
        //}

        forceinline FP pairGenerator(FP chi, FP randomNumber)
        {
            if (randomNumber < (FP)0.5)
                return breit_wheeler.inv_cdf(randomNumber, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - randomNumber, chi);
        }

        void operator()(ParticleProxy3d* particle, ValueField field, FP timeStep)
        {}

        void operator()(Particle3d* particle, ValueField field, FP timeStep)
        {
            ParticleProxy3d particleProxy(*particle);
            this->operator()(&particleProxy, field, timeStep);
        }

        Compton compton;
        Breit_wheeler breit_wheeler;

    private:

        //forceinline FP randomNumberOmp()
        //{
        //    int threadId = OMP_GET_THREAD_NUM();
        //    return distribution[threadId](randGenerator[threadId]);
        //}

        forceinline FP randomNumberOmp(const int threadId)
        {
            //int threadId = OMP_GET_THREAD_NUM();
            return distribution[threadId](randGenerator[threadId]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        const FP schwingerField;

        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED<YeeGrid> Scalar_Fast_QED_Yee;
    typedef Scalar_Fast_QED<PSTDGrid> Scalar_Fast_QED_PSTD;
    typedef Scalar_Fast_QED<PSATDGrid> Scalar_Fast_QED_PSATD;
    typedef Scalar_Fast_QED<AnalyticalField> Scalar_Fast_QED_Analytical;
}
