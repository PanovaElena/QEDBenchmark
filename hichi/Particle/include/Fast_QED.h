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

#include <cstdint>
#include <omp.h>
#include <random>

using namespace constants;
namespace pfc
{
    template <class TGrid>  // may be AnalyticalField or any Grid type
    class Scalar_Fast_QED : public ParticlePusher
    {
    public:

        Scalar_Fast_QED() : compton(), breit_wheeler()
        {
            schwingerField = sqr(Constants<FP>::electronMass() * Constants<FP>::lightVelocity())
                * Constants<FP>::lightVelocity() / (-Constants<FP>::electronCharge() * Constants<FP>::planck());

            int max_threads = OMP_GET_MAX_THREADS();

            randGenerator.resize(max_threads);
            distribution.resize(max_threads);
            for (int i = 0; i < max_threads; i++)
                distribution[i] = std::uniform_real_distribution<FP>(0.0, 1.0);

            this->afterAvalanchePhotons.resize(max_threads);
            this->afterAvalancheParticles.resize(max_threads);
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

        void processParticles(Ensemble3d* particles, TGrid* grid, FP timeStep)
        {
            int max_threads = OMP_GET_MAX_THREADS();

            if ((*particles)[Photon].size())
                HandlePhotons((*particles)[Photon], grid, timeStep);
            if ((*particles)[Electron].size())
                HandleParticles((*particles)[Electron], grid, timeStep);
            if ((*particles)[Positron].size())
                HandleParticles((*particles)[Positron], grid, timeStep);

            for (int th = 0; th < max_threads; th++)
            {
                for (int ind = 0; ind < this->afterAvalanchePhotons[th].size(); ind++)
                {
                    particles->addParticle(this->afterAvalanchePhotons[th][ind]);
                }
                for (int ind = 0; ind < this->afterAvalancheParticles[th].size(); ind++)
                {
                    particles->addParticle(this->afterAvalancheParticles[th][ind]);
                }

                this->afterAvalanchePhotons[th].clear();
                this->afterAvalancheParticles[th].clear();
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

        void HandlePhotons(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {
#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                int thread_id = OMP_GET_THREAD_NUM();

                HandlePhoton(particles[i], e, b, timeStep,
                    this->afterAvalancheParticles[thread_id],
                    this->afterAvalanchePhotons[thread_id]);
            }

            particles.clear();
        }

        void HandleParticles(ParticleArray3d& particles, TGrid* grid, FP timeStep)
        {
#pragma omp parallel for
            for (int i = 0; i < particles.size(); i++)
            {
                FP3 pPos = particles[i].getPosition();
                FP3 e, b;

                e = grid->getE(pPos);
                b = grid->getB(pPos);

                int thread_id = OMP_GET_THREAD_NUM();

                HandleParticle(particles[i], e, b, timeStep,
                    this->afterAvalancheParticles[thread_id],
                    this->afterAvalanchePhotons[thread_id]);
            }
        }

        void HandlePhoton(Particle3d& particle, const FP3& e, const FP3& b, FP timeStep,
            std::vector<Particle3d>& afterAvalancheParticles, std::vector<Particle3d>& afterAvalanchePhotons)
        {
            std::vector<Particle3d> avalancheParticles;
            std::vector<FP> timeAvalancheParticles;
            std::vector<Particle3d> avalanchePhotons;
            std::vector<FP> timeAvalanchePhotons;

            avalanchePhotons.push_back(particle);
            timeAvalanchePhotons.push_back((FP)0.0);

            RunAvalanche(e, b, timeStep,
                avalancheParticles, timeAvalancheParticles,
                avalanchePhotons, timeAvalanchePhotons);

            for (int k = 0; k != avalanchePhotons.size(); k++)
                if (avalanchePhotons[k].getGamma() != (FP)1.0)
                    afterAvalanchePhotons.push_back(avalanchePhotons[k]);

            for (int k = 0; k != avalancheParticles.size(); k++)
                afterAvalancheParticles.push_back(avalancheParticles[k]);
        }

        void HandleParticle(Particle3d& particle, const FP3& e, const FP3& b, FP timeStep,
            std::vector<Particle3d>& afterAvalancheParticles, std::vector<Particle3d>& afterAvalanchePhotons)
        {
            std::vector<Particle3d> avalancheParticles;
            std::vector<FP> timeAvalancheParticles;
            std::vector<Particle3d> avalanchePhotons;
            std::vector<FP> timeAvalanchePhotons;

            avalancheParticles.push_back(particle);
            timeAvalancheParticles.push_back((FP)0.0);

            RunAvalanche(e, b, timeStep,
                avalancheParticles, timeAvalancheParticles,
                avalanchePhotons, timeAvalanchePhotons);

            for (int k = 0; k != avalanchePhotons.size(); k++)
                if (avalanchePhotons[k].getGamma() != (FP)1.0)
                    afterAvalanchePhotons.push_back(avalanchePhotons[k]);

            particle.setMomentum(avalancheParticles[0].getMomentum());
            particle.setPosition(avalancheParticles[0].getPosition());
            for (int k = 1; k != avalancheParticles.size(); k++)
                afterAvalancheParticles.push_back(avalancheParticles[k]);
        }
        
        void RunAvalanche(const FP3& E, const FP3& B, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {         
            int countProcessedParticles = 0;
            int countProcessedPhotons = 0;

            while (countProcessedParticles != avalancheParticles.size()
                || countProcessedPhotons != avalanchePhotons.size())
            {
                for (int k = countProcessedParticles; k != avalancheParticles.size(); k++)
                {
                    oneParticleStep(avalancheParticles[k], E, B, timeAvalancheParticles[k], timeStep,
                        avalanchePhotons, timeAvalanchePhotons);
                    countProcessedParticles++;
                }
                for (int k = countProcessedPhotons; k != avalanchePhotons.size(); k++)
                {
                    onePhotonStep(avalanchePhotons[k], E, B, timeAvalanchePhotons[k], timeStep,
                        avalancheParticles, timeAvalancheParticles);
                    countProcessedPhotons++;
                }
            }
        }

        void oneParticleStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, FP timeStep,
            std::vector<Particle3d>& avalanchePhotons, std::vector<FP>& timeAvalanchePhotons)
        {
            while (time < timeStep)
            {
                FP3 v = particle.getVelocity();

                FP H_eff = sqr(E + (1 / Constants<FP>::lightVelocity()) * VP(v, B))
                    - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                if (H_eff < 0) H_eff = 0;
                H_eff = sqrt(H_eff);
                FP gamma = particle.getGamma();
                FP chi = gamma * H_eff / schwingerField;
                FP rate = 0.0, dt = (FP)2*timeStep;
                if (chi > 0.0 && this->photonEmissionEnabled)
                {
                    rate = compton.rate(chi);
                    dt = getDtParticle(particle, rate, chi);
                }

                if (dt + time > timeStep)
                {
                    Boris(particle, E, B, timeStep - time);
                    time = timeStep;
                }
                else
                {
                    Boris(particle, E, B, dt);
                    time += dt;

                    FP3 v = particle.getVelocity();
                    FP H_eff = sqr(E + (1 / Constants<FP>::lightVelocity()) * VP(v, B))
                        - sqr(SP(E, v) / Constants<FP>::lightVelocity());
                    if (H_eff < 0) H_eff = 0;
                    H_eff = sqrt(H_eff);
                    FP gamma = particle.getGamma();
                    FP chi_new = gamma * H_eff / schwingerField;

                    FP delta = photonGenerator((chi + chi_new)/(FP)2.0);
                    
                    Particle3d newParticle;
                    newParticle.setType(Photon);
                    newParticle.setWeight(particle.getWeight());
                    newParticle.setPosition(particle.getPosition());
                    newParticle.setMomentum(delta * particle.getMomentum());

                    avalanchePhotons.push_back(newParticle);
                    timeAvalanchePhotons.push_back(time);
                    particle.setMomentum((1 - delta) * particle.getMomentum());
                }
            }
        }

        void onePhotonStep(Particle3d& particle, const FP3& E, const FP3& B, FP& time, FP timeStep,
            std::vector<Particle3d>& avalancheParticles, std::vector<FP>& timeAvalancheParticles)
        {
            FP3 k_ = particle.getVelocity();
            k_ = ((FP)1 / k_.norm()) * k_; // normalized wave vector
            FP H_eff = sqrt(sqr(E + VP(k_, B)) - sqr(SP(E, k_)));
            FP gamma = particle.getMomentum().norm()
                / (Constants<FP>::electronMass() * Constants<FP>::lightVelocity());
            FP chi = gamma * H_eff / schwingerField;

            FP rate = 0.0, dt = (FP)2 * timeStep;
            if (chi > 0.0 && this->pairProductionEnabled)
            {
                rate = breit_wheeler.rate(chi);
                dt = getDtPhoton(particle, rate, chi, gamma);
            }

            if (dt + time > timeStep)
            {
                particle.setPosition(particle.getPosition()
                    + (timeStep - time) * Constants<FP>::lightVelocity() * k_);
                time = timeStep;
            }
            else
            {
                particle.setPosition(particle.getPosition()
                    + dt * Constants<FP>::lightVelocity() * k_);
                time += dt;
                FP delta = pairGenerator(chi);

                Particle3d newParticle;
                newParticle.setType(Electron);
                newParticle.setWeight(particle.getWeight());
                newParticle.setPosition(particle.getPosition());
                newParticle.setMomentum(delta * particle.getMomentum());

                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                newParticle.setType(Positron);
                newParticle.setMomentum(((FP)1 - delta) * particle.getMomentum());
                avalancheParticles.push_back(newParticle);
                timeAvalancheParticles.push_back(time);

                particle.setP((FP)0.0 * particle.getP());  // ???
                time = timeStep;
            }
        }
        
        FP getDtParticle(Particle3d& particle, FP rate, FP chi)
        {
            FP r = -log(random_number_omp());
            r *= (particle.getGamma()) / (chi);
            return r / rate;
        }
        
        FP photonGenerator(FP chi)
        {
            FP r = random_number_omp();
            return compton.inv_cdf(r, chi);
        }


        FP getDtPhoton(Particle3d& particle, FP rate, FP chi, FP gamma)
        {
            FP r = -log(random_number_omp());
            r *= (gamma) / (chi);
            return r / rate;
        }
        
        FP pairGenerator(FP chi)
        {
            FP r = random_number_omp();
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

        FP random_number_omp()
        {
            int thread_id = OMP_GET_THREAD_NUM();
            return distribution[thread_id](randGenerator[thread_id]);
        }

        std::vector<std::default_random_engine> randGenerator;
        std::vector<std::uniform_real_distribution<FP>> distribution;

        std::vector<std::vector<Particle3d>> afterAvalanchePhotons, afterAvalancheParticles;

        FP schwingerField;
        
        bool photonEmissionEnabled = true, pairProductionEnabled = true;
    };

    typedef Scalar_Fast_QED<YeeGrid> Scalar_Fast_QED_Yee;
    typedef Scalar_Fast_QED<PSTDGrid> Scalar_Fast_QED_PSTD;
    typedef Scalar_Fast_QED<PSATDGrid> Scalar_Fast_QED_PSATD;
    typedef Scalar_Fast_QED<AnalyticalField> Scalar_Fast_QED_Analytical;
}
