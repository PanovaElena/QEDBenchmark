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

        template <class T>
        struct VectorOfGenElements {
            constexpr static int factor = 2;

            int n = 0, capacity = 8;
            std::vector<T> data;

            VectorOfGenElements() : data(capacity) {}

            T& operator[](int i) { return data[i]; }
            const T& operator[](int i) const { return data[i]; }

            const int& size() const { return n; }
            void clear() { n = 0; }

            void push_back(const T& value) {
                if (n == capacity) {
                    capacity *= factor;
                    data.resize(capacity);
                }
                data[n++] = value;
            }
        
            void extend(const VectorOfGenElements& v, int begin, int end) {
                int num = end - begin;
                if (num <= 0) return;
                if (n + num >= capacity) {
                    while (n + num >= capacity)
                        capacity *= factor;
                    data.resize(capacity);
                }
                std::memmove(data.data() + n, v.data.data() + begin, num * sizeof(T));
                n += num;
            }

        };

        Scalar_Fast_QED() : compton(), breit_wheeler(),
            schwingerField(sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
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
                HandlePhotons((*particles)[Photon], e[0], b[0], timeStep, generatedParticles, generatedPhotons);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], e[1], b[1], timeStep, generatedParticles, generatedPhotons);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], e[2], b[2], timeStep, generatedParticles, generatedPhotons);

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
        void Boris(Particle3d&& particle, const FP3& e, const FP3& b, FP timeStep)
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

        void Boris(ParticleProxy3d&& particle, const FP3& e, const FP3& b, FP timeStep)
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

        void HandlePhotons(ParticleArray3d& photons,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            std::vector<VectorOfGenElements<Particle3d>>& generatedParticles,
            std::vector<VectorOfGenElements<Particle3d>>& generatedPhotons)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<VectorOfGenElements<Particle3d>> avalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP>> timeAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<Particle3d>> avalanchePhotonsThread(maxThreads);
            std::vector<VectorOfGenElements<FP>> timeAvalanchePhotonsThread(maxThreads);

#pragma omp parallel for
            for (int i = 0; i < photons.size(); i++)
            {
                FP3 e = ve[i], b = vb[i];

                int threadId = OMP_GET_THREAD_NUM();

                VectorOfGenElements<Particle3d>& avalancheParticles = avalancheParticlesThread[threadId];
                VectorOfGenElements<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[threadId];
                VectorOfGenElements<Particle3d>& avalanchePhotons = avalanchePhotonsThread[threadId];
                VectorOfGenElements<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[threadId];

                RunAvalanchePhoton(photons[i], e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                int lastIndex = 0;
                for (int k = 0; k < avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) {
                        if (lastIndex != k)
                            std::swap(avalanchePhotons[k], avalanchePhotons[lastIndex]);
                        lastIndex++;
                    }
                generatedPhotons[threadId].extend(avalanchePhotons, 0, lastIndex);

                // push generated particles
                generatedParticles[threadId].extend(avalancheParticles, 0, avalancheParticles.size());

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void HandleParticles(ParticleArray3d& particles,
            const std::vector<FP3>& ve, const std::vector<FP3>& vb, FP timeStep,
            std::vector<VectorOfGenElements<Particle3d>>& generatedParticles,
            std::vector<VectorOfGenElements<Particle3d>>& generatedPhotons)
        {
            int maxThreads = OMP_GET_MAX_THREADS();

            std::vector<VectorOfGenElements<Particle3d>> avalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<FP>> timeAvalancheParticlesThread(maxThreads);
            std::vector<VectorOfGenElements<Particle3d>> avalanchePhotonsThread(maxThreads);
            std::vector<VectorOfGenElements<FP>> timeAvalanchePhotonsThread(maxThreads);

#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 e = ve[i], b = vb[i];

                int threadId = OMP_GET_THREAD_NUM();

                VectorOfGenElements<Particle3d>& avalancheParticles = avalancheParticlesThread[threadId];
                VectorOfGenElements<FP>& timeAvalancheParticles = timeAvalancheParticlesThread[threadId];
                VectorOfGenElements<Particle3d>& avalanchePhotons = avalanchePhotonsThread[threadId];
                VectorOfGenElements<FP>& timeAvalanchePhotons = timeAvalanchePhotonsThread[threadId];

                RunAvalancheParticle(particles[i], e, b, timeStep,
                    avalancheParticles, timeAvalancheParticles,
                    avalanchePhotons, timeAvalanchePhotons);

                // push generated physical photons
                int lastIndex = 0;
                for (int k = 0; k < avalanchePhotons.size(); k++)
                    if (avalanchePhotons[k].getGamma() != (FP)1.0) {
                        if (lastIndex != k)
                            std::swap(avalanchePhotons[k], avalanchePhotons[lastIndex]);
                        lastIndex++;
                    }
                generatedPhotons[threadId].extend(avalanchePhotons, 0, lastIndex);

                // save new position and momentum of the last particle
                particles[i].setMomentum(avalancheParticles[0].getMomentum());
                particles[i].setPosition(avalancheParticles[0].getPosition());
                // push generated particles
                generatedParticles[threadId].extend(avalancheParticles, 1, avalancheParticles.size());

                avalancheParticles.clear();
                timeAvalancheParticles.clear();
                avalanchePhotons.clear();
                timeAvalanchePhotons.clear();
            }
        }

        void RunAvalancheParticle(const Particle3d& particle, const FP3& e, const FP3& b, FP timeStep,
            VectorOfGenElements<Particle3d>& avalancheParticles, VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<Particle3d>& avalanchePhotons, VectorOfGenElements<FP>& timeAvalanchePhotons)
        {
            // start avalanche with current particle
            avalancheParticles.push_back(particle);
            timeAvalancheParticles.push_back((FP)0.0);

            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            // now countProcessedPhotons == avalanchePhotons.size()
            while (countProcessedParticles != avalancheParticles.size())
            {
                for (int k = countProcessedParticles; k != avalancheParticles.size(); k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                }
                countProcessedParticles = avalancheParticles.size();

                for (int k = countProcessedPhotons; k != avalanchePhotons.size(); k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                }
                countProcessedPhotons = avalanchePhotons.size();
            }
        }

        void RunAvalanchePhoton(const Particle3d& photon, const FP3& e, const FP3& b, FP timeStep,
            VectorOfGenElements<Particle3d>& avalancheParticles, VectorOfGenElements<FP>& timeAvalancheParticles,
            VectorOfGenElements<Particle3d>& avalanchePhotons, VectorOfGenElements<FP>& timeAvalanchePhotons)
        {
            // start avalanche with current photon
            avalanchePhotons.push_back(photon);
            timeAvalanchePhotons.push_back((FP)0.0);

            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            while (countProcessedPhotons != avalanchePhotons.size())
            {
                for (int k = countProcessedPhotons; k != avalanchePhotons.size(); k++)
                {
                    onePhotonStep(avalanchePhotons[k], e, b, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                }
                countProcessedPhotons = avalanchePhotons.size();

                for (int k = countProcessedParticles; k != avalancheParticles.size(); k++)
                {
                    oneParticleStep(avalancheParticles[k], e, b, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                }
                countProcessedParticles = avalancheParticles.size();
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& e, const FP3& b, FP& time, FP timeStep,
            VectorOfGenElements<Particle3d>& avalanchePhotons, VectorOfGenElements<FP>& timeAvalanchePhotons)
        {
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();

                FP H_eff = sqr(e + ((FP)1 / Constants<FP>::lightVelocity()) * VP(v, b))
                    - sqr(SP(e, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff / this->schwingerField;
                FP rate = (FP)0.0, dt = (FP)2*timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDt(rate, chi, gamma);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, e, b, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, e, b, dt);
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

                    avalanchePhotons.push_back(newPhoton);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& photon, const FP3& e, const FP3& b, FP& time, FP timeStep,
            VectorOfGenElements<Particle3d>& avalancheParticles, VectorOfGenElements<FP>& timeAvalancheParticles)
        {
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

                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * photon.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                photon.setP((FP)0.0 * photon.getP());  // mark old photon as deleted
                time = timeStep;
            }
        }
        
        FP getDt(FP rate, FP chi, FP gamma)
        {
            FP r = -log(randomNumberOmp());
            r *= gamma / chi;
            return r / rate;
        }
        
        FP photonGenerator(FP chi)
        {
            FP r = randomNumberOmp();
            return compton.inv_cdf(r, chi);
        }
        
        FP pairGenerator(FP chi)
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

        FP randomNumberOmp()
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
