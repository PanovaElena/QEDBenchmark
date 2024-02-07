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

        static const int chunkSize = 8;

        template <class T>
        struct VectorOfGenElements {
            constexpr static int factor = 2;

            int n = 0, capacity = 8;
            std::vector<T> data;

            VectorOfGenElements() : data(capacity) {}

            forceinline T& operator[](int i) { return data[i]; }
            forceinline const T& operator[](int i) const { return data[i]; }

            forceinline const int& size() const { return n; }
            forceinline void clear() { n = 0; }

            forceinline void push_back(const T& value) {
                if (n == capacity) {
                    capacity *= factor;
                    data.resize(capacity);
                }
                data[n++] = value;
            }

            forceinline void decreaseSize(int newSize) { n = newSize; }
            forceinline void pop_back() { n--; }

            forceinline void removeElement(int index) {
                if (index < n - 1)
                    std::swap(data[index], data[n - 1]);
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
                    begin, end
                );
            }
            if (chunkRem != 0)
                handlePhotonChunk(photons, ve, vb, timeStep,
                    generatedParticlesThread[0], generatedPhotonsThread[0],
                    timeAvalancheParticlesThread[0], timeAvalanchePhotonsThread[0],
                    eAvalancheParticlesThread[0], bAvalancheParticlesThread[0],
                    eAvalanchePhotonsThread[0], bAvalanchePhotonsThread[0],
                    n - chunkRem, n
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
                    begin, end
                );
            }
            if (chunkRem != 0)
                handleParticleChunk(particles, ve, vb, timeStep,
                    generatedParticlesThread[0], generatedPhotonsThread[0],
                    timeAvalancheParticlesThread[0], timeAvalanchePhotonsThread[0],
                    eAvalancheParticlesThread[0], bAvalancheParticlesThread[0],
                    eAvalanchePhotonsThread[0], bAvalanchePhotonsThread[0],
                    n - chunkRem, n
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
            const int begin, const int end
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
                prevGeneratedParticlesSize, prevGeneratedPhotonsSize);

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
            const int begin, const int end
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
                prevGeneratedParticlesSize, prevGeneratedPhotonsSize);

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
            int generatedParticlesStart, int generatedPhotonsStart)
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
                        countProcessedParticles + chunkNum * chunkSize, chunkSize, generatedParticlesStart
                    );
                if (chunkRem != 0)
                    oneParticleStep(generatedParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        timeAvalancheParticles, timeStep,
                        generatedPhotons, timeAvalanchePhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        countProcessedParticles + n - chunkRem, chunkRem, generatedParticlesStart
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
                        countProcessedPhotons + chunkNum * chunkSize, chunkSize, generatedPhotonsStart
                    );
                if (chunkRem != 0)
                    onePhotonStep(generatedPhotons,
                        eAvalanchePhotons, bAvalanchePhotons,
                        timeAvalanchePhotons, timeStep,
                        generatedParticles, timeAvalancheParticles,
                        eAvalancheParticles, bAvalancheParticles,
                        countProcessedPhotons + n - chunkRem, chunkRem, generatedPhotonsStart
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
            int begin, int size, int auxShift)
        {
            const int end = begin + size;
            begin = sortHandledParticles(particles, timeStep, vtime, ve, vb, begin, end, auxShift);

            while (begin < end) {
#pragma omp simd
                for (int i = 0; i < end - begin; i++) {

                    Particle3d& particle = particles[begin + i];
                    FP& time = vtime[begin + i - auxShift];
                    FP3& e = ve[begin + i - auxShift];
                    FP3& b = vb[begin + i - auxShift];

                    FP3 v = particle.getVelocity();

                    FP H_eff = sqr(e + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b))
                        - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi = gamma * H_eff / this->schwingerField;
                    FP rate = (FP)0.0, dt = (FP)2 * timeStep;
                    if (chi > 0.0 && this->photonEmissionEnabled)
                    {
                        rate = compton.rate(chi);
                        dt = getDt(rate, chi, gamma);
                    }

                    if (dt + time > timeStep)
                    {
                        boris(particle, e, b, timeStep - time);
                        time = timeStep;
                    }
                    else
                    {
                        boris(particle, e, b, dt);
                        time += dt;

                        FP3 v = particle.getVelocity();
                        FP H_eff = sqr(e + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b))
                            - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                        if (H_eff < 0) H_eff = 0;
                        H_eff = sqrt(H_eff);
                        FP gamma = particle.getGamma();
                        FP chi_new = gamma * H_eff / this->schwingerField;

                        FP delta = photonGenerator((chi + chi_new) / (FP)2.0);

                        Particle3d newPhoton;
                        newPhoton.setType(Photon);
                        newPhoton.setWeight(particle.getWeight());
                        newPhoton.setPosition(particle.getPosition());
                        newPhoton.setMomentum(delta * particle.getMomentum());

                        generatedPhotons.push_back(newPhoton);
                        timeAvalanchePhotons.push_back(time);
                        eAvalanchePhotons.push_back(e);
                        bAvalanchePhotons.push_back(b);

                        particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                    }
                }
                
                begin = sortHandledParticles(particles, timeStep, vtime, ve, vb, begin, end, auxShift);
            }
        }

        void onePhotonStep(VectorOfGenElements<Particle3d>& photons,
            VectorOfGenElements<FP3>& ve, VectorOfGenElements<FP3>& vb,
            VectorOfGenElements<FP>& vtime, FP timeStep,
            VectorOfGenElements<Particle3d>& generatedParticles,
            VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<FP3>& eAvalancheParticles,
            VectorOfGenElements<FP3>& bAvalancheParticles,
            int begin, int size, int auxShift)
        {
#pragma omp simd
            for (int i = 0; i < size; i++) {

                Particle3d& photon = photons[begin + i];
                FP& time = vtime[begin + i - auxShift];
                FP3& e = ve[begin + i - auxShift];
                FP3& b = vb[begin + i - auxShift];

                FP3 k_ = photon.getVelocity();
                k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
                FP H_eff = sqrt(sqr(e + VP(k_, b)) - sqr(SP(e, k_)));
                FP gamma = photon.getMomentum().norm()
                    / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
                FP chi = gamma * H_eff / this->schwingerField;

                FP rate = 0.0, dt = (FP)2 * timeStep;
                if (chi > 0.0 && this->pairProductionEnabled)
                {
                    rate = breit_wheeler.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }

                if (dt + time > timeStep)
                {
                    photon.setPosition(photon.getPosition()
                        + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                    time = timeStep;
                }
                else
                {
                    photon.setPosition(photon.getPosition()
                        + dt * Constants<FP>::lightVelocity() * k_);
                    time += dt;
                    FP delta = pairGenerator(chi);

                    Particle3d newParticle;
                    newParticle.setType(Electron);
                    newParticle.setWeight(photon.getWeight());
                    newParticle.setPosition(photon.getPosition());
                    newParticle.setMomentum(delta * photon.getMomentum());

                    generatedParticles.push_back(newParticle);
                    timeAvalancheParticles.push_back(time);
                    eAvalancheParticles.push_back(e);
                    bAvalancheParticles.push_back(b);

                    newParticle.setType(Positron);
                    newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());

                    generatedParticles.push_back(newParticle);
                    timeAvalancheParticles.push_back(time);
                    eAvalancheParticles.push_back(e);
                    bAvalancheParticles.push_back(b);

                    photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                    time = timeStep;
                }
            }
        }

        forceinline int sortHandledParticles(VectorOfGenElements<Particle3d>& particles, FP timeStep,
            VectorOfGenElements<FP>& vtime, VectorOfGenElements<FP3>& ve, VectorOfGenElements<FP3>& vb,
            int begin, int end, int auxShift)
        {
            int lastIndex = begin;
            for (int i = begin; i < end; i++)
                if (vtime[i - auxShift] >= timeStep) {
                    if (lastIndex != i) {
                        std::swap(particles[i], particles[lastIndex]);
                        std::swap(vtime[i - auxShift], vtime[lastIndex - auxShift]);
                        std::swap(ve[i - auxShift], ve[lastIndex - auxShift]);
                        std::swap(vb[i - auxShift], vb[lastIndex - auxShift]);
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
            generatedPhotons.decreaseSize(lastIndex);
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
            generatedParticles.decreaseSize(generatedParticles.size() - size);
        }

        forceinline FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(randomNumberOmp());
            r *= gamma / chi;
            return r / rate;
        }

        forceinline FP photonGenerator(FP chi)
        {
            FP r = randomNumberOmp();
            return compton.inv_cdf(r, chi);
        }

        forceinline FP pairGenerator(FP chi)
        {
            FP r = randomNumberOmp();
            if (r < (FP)0.5)
                return breit_wheeler.inv_cdf(r, chi);
            else
                return (FP)1.0 - breit_wheeler.inv_cdf((FP)1.0 - r, chi);
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

        forceinline FP randomNumberOmp()
        {
            int threadId = OMP_GET_THREAD_NUM();
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
